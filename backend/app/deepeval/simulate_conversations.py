"""Step 2 (graph-direct) - Turn the goldens from step 1 into real multi-turn
conversations by having DeepEval's ConversationSimulator drive the RAG LangGraph
DIRECTLY (no FastAPI, no /answer endpoint, no SSE parsing).

This is the drop-in replacement for `simulate_conversations.py` for people who do
not want to stand up the HTTP server during dataset generation. It builds the
exact same graph `main.py` builds at startup (same embedder, Qdrant client,
PathImageChatOllama, CohereReranker, rewrite LLM) and calls `graph.ainvoke(...)`
from inside DeepEval's `model_callback`.

How a turn is simulated
-----------------------
DeepEval's `ConversationSimulator` plays the user (an LLM that follows each
golden's scenario). For every user message it calls `model_callback(input,
thread_id)`, which here:
  1. looks up the in-memory history for that thread (one list per conversation);
  2. invokes the compiled RAG graph with that history (rewrite -> retrieve ->
     rerank -> generate), wrapped in `propagate_attributes(...)` + a Langfuse
     `CallbackHandler` so every turn shows up as a trace in the Langfuse UI,
     grouped into one session per conversation;
  3. appends the (user, assistant) exchange to the history so the next turn's
     rewrite sees it;
  4. returns a `Turn(role="assistant", content=answer, metadata={...})`.

The graph is the same one `controller.py` runs for `/answer`, minus the SSE
streaming, the Postgres persistence and `compact_history` (this script keeps
history purely in memory - conversations are short, so the trigger is never hit).
Nothing is written to Postgres. Output is one CSV row per turn, appended after
each finished conversation (so a crash loses nothing and re-running resumes).

Verified against deepeval `main` (see notes at the bottom of this file).

Example
-------
    # smoke test on 3 goldens
    python simulate_conversations_graph.py --goldens conversational_goldens.csv --limit 3

    # the real run (resumes automatically)
    python simulate_conversations_graph.py --goldens conversational_goldens.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from collections import defaultdict

# Heavy / project imports are deferred into functions so `--help` works without
# the app environment being importable.
# --- sys.path bootstrap ---------------------------------------------------- #
# Make the script run regardless of how it's invoked:
#   python app/deepeval/simulate_conversations.py   (cwd=backend/)
#   python -m app.deepeval.simulate_conversations   (cwd=backend/)
#   python simulate_conversations.py                (cwd=app/deepeval/)
# Two paths are needed because the script imports from two roots:
#   - from app.graph...        -> needs backend/ on sys.path
#   - from gen_conversational_goldens import build_llm -> needs app/deepeval/ on sys.path
from pathlib import Path as _Path
from typing import Any, Dict, List, Optional, cast

_HERE = _Path(__file__).resolve().parent  # backend/app/deepeval/
_REPO_ROOT = _HERE.parent.parent  # backend/
for _p in (_HERE, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_TOP_K = 5  # matches controller.py

OUT_COLUMNS = [
    "golden_id",
    "turn_index",
    "role",
    "content",
    "ragwise_conversation_id",
    "retrieved_chunk_ids",
    "retrieved_paper_ids",
    "retrieved_contexts",
    "latency_s",
    "rewritten_query",
    "rerank_fallback",
    "images_used",
    "history_len",
]


# --------------------------------------------------------------------------- #
# Graph / component construction (mirrors main.py lifespan exactly)
# --------------------------------------------------------------------------- #
def build_graph(args):
    """Return the compiled RAG graph + the Qdrant client (to close on shutdown).

    Every constructor call here is copied verbatim from `main.py`'s lifespan so
    the dataset you generate is produced by the same code path as production.
    """
    from langchain_ollama import ChatOllama
    from langfuse import Langfuse
    from qdrant_client import QdrantClient

    from app.config.config import settings
    from app.graph.rag_graph import build_rag_graph
    from app.services.rerank import CohereReranker
    from app.services.retrieve_from_qdrant import QueryEmbedder
    from app.services.vision_llm import PathImageChatOllama

    # Langfuse must be initialised before any propagate_attributes / CallbackHandler
    # call, otherwise traces are silently dropped. Skip entirely when --no-trace.
    if args.trace:
        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url,
        )
        print("Langfuse tracing: ON")
    else:
        print("Langfuse tracing: OFF (--no-trace)")

    print("Loading BGE-M3 embedder...")
    embedder = QueryEmbedder()
    embedder.load()

    print("Connecting to Qdrant...")
    client = QdrantClient(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY"),
        prefer_grpc=True,
        grpc_port=6334,
    )

    print(f"Initializing graph LLM (PathImageChatOllama, {args.graph_model})...")
    llm = PathImageChatOllama(
        model=args.graph_model,
        temperature=args.graph_temperature,
        num_ctx=args.graph_num_ctx,
    )

    print("Initializing rewrite LLM...")
    rewrite_llm = ChatOllama(
        model=args.graph_model,
        temperature=0,
        num_ctx=4096,
        num_predict=128,
        reasoning=False,
    )

    print("Initializing Cohere reranker...")
    reranker = CohereReranker(
        api_key=settings.cohere_api_key,
        model=settings.cohere_rerank_model,
    )

    print(f"Building RAG graph (use_images={args.use_images})...")
    graph = build_rag_graph(
        embedder=embedder,
        client=client,
        llm=llm,
        reranker=reranker,
        use_images=args.use_images,
        rewrite_llm=rewrite_llm,
    )
    return graph, client


# --------------------------------------------------------------------------- #
# Goldens loading / resume (same shape as the HTTP step 2)
# --------------------------------------------------------------------------- #
def done_golden_ids(out_path: str) -> set:
    if not os.path.exists(out_path):
        return set()
    with open(out_path, newline="", encoding="utf-8") as f:
        return {r["golden_id"] for r in csv.DictReader(f)}


def load_goldens(path: str, skip_ids: set, limit: Optional[int]):
    from deepeval.dataset import ConversationalGolden

    goldens = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gid = row["golden_id"]
            if gid in skip_ids:
                continue
            goldens.append(
                ConversationalGolden(
                    scenario=row["scenario"],
                    expected_outcome=row.get("expected_outcome") or None,
                    context=json.loads(row["context_chunks"])
                    if row.get("context_chunks")
                    else None,
                    name=gid,
                    user_description=None,
                    multimodal=False,
                    additional_metadata={
                        "used_source_files": json.loads(
                            row.get("used_source_files") or "[]"
                        ),
                    },
                )
            )
            if limit and len(goldens) >= limit:
                break
    return goldens


# --------------------------------------------------------------------------- #
# Core: build the simulator with a graph-direct model_callback and run it
# --------------------------------------------------------------------------- #
def run(args) -> None:
    from deepeval.simulator import ConversationSimulator
    from deepeval.test_case import Turn

    # Reuse the simulated-user LLM builder from step 1 (DeepEval OllamaModel).
    from gen_conversational_goldens import build_llm
    from langfuse import propagate_attributes
    from langfuse.langchain import CallbackHandler

    from app.graph.rag_graph import GenerationError, RetrievalError

    graph, qdrant_client = build_graph(args)

    skip = done_golden_ids(args.out)
    goldens = load_goldens(args.goldens, skip, args.limit)
    if skip:
        print(f"Resuming: {len(skip)} conversations already in {args.out}")
    if not goldens:
        print("Nothing to simulate.")
        try:
            qdrant_client.close()
        except Exception:
            pass
        return
    print(
        f"Simulating {len(goldens)} conversations "
        f"(max {args.max_user_turns} user turns each, concurrency={args.concurrency}) "
        f"via the RAG graph directly."
    )

    # Per-thread state. thread_id is generated by the simulator (one uuid per
    # conversation) and is stable across all turns of that conversation.
    conv_histories: Dict[str, List[dict]] = defaultdict(
        list
    )  # thread_id -> [{"role","content"}]
    details: Dict[str, List[dict]] = defaultdict(
        list
    )  # thread_id -> per-assistant-turn details

    # Cohere rate limiter: 10 RPM cap means at most 1 call every 6s.
    # Without this, fast turns (3.5s end-to-end) would fire ~17 rerank
    # calls/min and trigger 429 cascades. List wrapping lets the closure
    # mutate the timestamp without `nonlocal`.
    MIN_RERANK_INTERVAL_S = 6.1
    last_rerank_time: List[float] = [0.0]

    async def model_callback(input: str, thread_id: str) -> Turn:
        # Enforce Cohere's 10-RPM cap: sleep the remainder of the 6.1s
        # window since the last turn started. If the previous turn took
        # longer than 6.1s (slow generation), no wait. If it was faster,
        # we sleep the remainder. Prevents 429 cascades.
        now = time.perf_counter()
        wait = MIN_RERANK_INTERVAL_S - (now - last_rerank_time[0])
        if wait > 0:
            await asyncio.sleep(wait)
        last_rerank_time[0] = time.perf_counter()

        history = conv_histories[thread_id]
        hist_len_before = len(history)  # what the graph's rewrite node will see
        t0 = time.perf_counter()

        state = None
        for attempt in range(args.retries + 1):
            try:
                input_state = {
                    "query": input,
                    "top_k": args.top_k,
                    "paper_id": None,
                    "has_picture": None,
                    "history": list(history),  # copy: graph must not mutate ours
                    "history_summary": None,  # in-memory mode: no compaction
                }
                if args.trace:
                    with propagate_attributes(
                        trace_name="rag-answer",
                        user_id=args.eval_user_id,
                        session_id=thread_id,
                        tags=["rag", "eval-sim"],
                        metadata={
                            "top_k": str(args.top_k),
                            "history_len": str(hist_len_before),
                            "use_images": str(args.use_images).lower(),
                            "turn": str(hist_len_before // 2 + 1),
                            "simulated": "true",
                        },
                    ):
                        state = await graph.ainvoke(
                            input_state,
                            config={"callbacks": [CallbackHandler()]},
                        )
                else:
                    state = await graph.ainvoke(input_state)
                break
            except (RetrievalError, GenerationError) as e:
                if attempt < args.retries:
                    wait = 2**attempt
                    print(
                        f"  [graph retry {attempt + 1}/{args.retries}] {type(e).__name__}: {e} - waiting {wait}s",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(wait)
                    continue
                # Final failure: surface an empty assistant turn so the conversation
                # is still flushed to CSV (with empty retrieval) instead of aborting
                # the whole batch.
                print(
                    f"  [graph failed] {thread_id}: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                state = {
                    "answer": "",
                    "chunks": [],
                    "rewritten_query": input,
                    "rerank_fallback": True,
                    "images_used": [],
                }
                break

        answer = (state or {}).get("answer") or ""
        chunks = (state or {}).get("chunks") or []

        # Append AFTER the call so the next turn's rewrite sees this exchange.
        history.append({"role": "user", "content": input})
        history.append({"role": "assistant", "content": answer})

        details[thread_id].append(
            {
                "chunk_ids": [c.get("chunk_id") for c in chunks],
                "paper_ids": [c.get("paper_id") for c in chunks],
                "contexts": [c.get("text") for c in chunks],
                "latency_s": round(time.perf_counter() - t0, 2),
                "rewritten_query": (state or {}).get("rewritten_query"),
                "rerank_fallback": bool((state or {}).get("rerank_fallback", False)),
                "images_used": [
                    {
                        "n": img.get("n"),
                        "chunk_id": img.get("chunk_id"),
                        "paper_id": img.get("paper_id"),
                        "path": img.get("path"),
                    }
                    for img in ((state or {}).get("images_used") or [])
                ],
                "history_len": hist_len_before,
            }
        )

        # metadata carries only the (small) thread id so on_complete can find the
        # details. Heavy retrieval data stays in `details` and is NEVER put on the
        # Turn, because the simulator serialises every Turn field into the
        # simulated-user prompt on later turns (see Turn.model_dump_for_prompt).
        return Turn(
            role="assistant",
            content=answer,
            metadata={"conversation_id": thread_id},
        )

    new_file = not os.path.exists(args.out)
    out_f = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=OUT_COLUMNS)
    if new_file:
        writer.writeheader()
        out_f.flush()

    def on_complete(test_case, _index: int) -> None:
        gid = test_case.name
        a_idx = 0
        for i, turn in enumerate(test_case.turns):
            row: Dict[str, Any] = {
                "golden_id": gid,
                "turn_index": i,
                "role": turn.role,
                "content": turn.content,
            }
            if turn.role == "assistant":
                conv_id = (turn.metadata or {}).get("conversation_id")
                d_list = details.get(conv_id, []) if isinstance(conv_id, str) else []
                d = d_list[a_idx] if a_idx < len(d_list) else {}
                a_idx += 1
                row.update(
                    ragwise_conversation_id=conv_id,
                    retrieved_chunk_ids=json.dumps(d.get("chunk_ids", [])),
                    retrieved_paper_ids=json.dumps(d.get("paper_ids", [])),
                    retrieved_contexts=json.dumps(
                        d.get("contexts", []), ensure_ascii=False
                    ),
                    latency_s=d.get("latency_s"),
                    rewritten_query=d.get("rewritten_query"),
                    rerank_fallback=d.get("rerank_fallback", False),
                    images_used=json.dumps(
                        d.get("images_used", []), ensure_ascii=False
                    ),
                    history_len=d.get("history_len"),
                )
            writer.writerow(row)
        out_f.flush()
        n_user = sum(1 for t in test_case.turns if t.role == "user")
        print(f"  saved {gid}: {n_user} user turns, {len(test_case.turns)} rows")

    simulator = ConversationSimulator(
        # DeepEval's public annotation still says Callable[[str], str], but its
        # simulator invokes async callbacks with input, turns, and thread_id and
        # expects a Turn at runtime. Keep the cast at this library boundary.
        model_callback=cast(Any, model_callback),
        simulator_model=build_llm(args),
        max_concurrent=args.concurrency,
    )
    try:
        simulator.simulate(
            conversational_goldens=goldens,
            max_user_simulations=args.max_user_turns,
            on_simulation_complete=on_complete,
        )
    except KeyboardInterrupt:
        print(
            "\nInterrupted; finished conversations are already in",
            args.out,
            file=sys.stderr,
        )
        raise
    except Exception as e:
        print(
            f"\nSimulation aborted: {type(e).__name__}: {e}\n"
            f"Finished conversations are already in {args.out}; re-run the same command to resume.",
            file=sys.stderr,
        )
        raise
    finally:
        out_f.close()
        try:
            qdrant_client.close()
        except Exception:
            pass

    print(f"Done -> {args.out}")


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--goldens",
        default="conversational_goldens.csv",
        help="CSV produced by gen_conversational_goldens.py (step 1).",
    )
    p.add_argument("--out", default="simulated_conversations_graph.csv")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only the first N not-yet-done goldens (smoke test).",
    )

    g = p.add_argument_group("Simulation")
    g.add_argument("--max-user-turns", type=int, default=3)
    g.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Conversations simulated in parallel. The shared embedder / "
        "reranker / LLM are concurrency-safe (FastAPI does the same).",
    )
    g.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry a turn's graph.ainvoke on RetrievalError/GenerationError.",
    )

    g = p.add_argument_group("Graph (RAG side - matches main.py)")
    g.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="top_k passed to the graph (controller uses 5).",
    )
    g.add_argument("--use-images", dest="use_images", action="store_true", default=True)
    g.add_argument(
        "--no-images",
        dest="use_images",
        action="store_false",
        help="Run text-only (use_images=False). Default is images on, like main.py.",
    )
    g.add_argument("--graph-model", default="gemma4:31b-cloud")
    g.add_argument("--graph-temperature", type=float, default=0.2)
    g.add_argument("--graph-num-ctx", type=int, default=16384)

    g = p.add_argument_group("Simulated user (DeepEval side - same as step 1)")
    g.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "gemma4:31b-cloud"),
        help="Simulated-user model (DeepEval OllamaModel).",
    )
    g.add_argument(
        "--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    )
    g.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Simulated-user temperature (a bit of variety).",
    )

    g = p.add_argument_group("Langfuse")
    g.add_argument(
        "--eval-user-id",
        default="eval-sim",
        help="Langfuse user_id stamped on every simulated turn's trace.",
    )
    g.add_argument(
        "--no-trace",
        dest="trace",
        action="store_false",
        default=True,
        help="Disable Langfuse tracing for this run.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
