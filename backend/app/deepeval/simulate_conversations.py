"""
Step 2 - Turn the goldens from step 1 into real multi-turn conversations by
having a simulated user talk to your running RAGwise `/answer` endpoint.

DeepEval's ConversationSimulator plays the user (an LLM following each golden's
scenario). For every user message it calls `model_callback`, which here POSTs to
/answer, reads the SSE stream, and returns the final answer as the assistant turn.
The server-side conversation_id is carried between turns, so your real
conversation loading / rewriting / summarisation code is what gets exercised.

IMPORTANT side effect: every simulated turn is saved by your backend into Postgres
(conversations + messages) under the user that owns the auth header you pass.
Use a dedicated eval user.

Output (appended after each finished conversation, so a crash loses nothing and
re-running resumes): one CSV row per turn.

Example
-------
    export RAGWISE_AUTH_HEADER="Authorization: Bearer <token of an eval user>"
    python simulate_conversations.py --goldens conversational_goldens.csv --limit 3   # smoke test
    python simulate_conversations.py --goldens conversational_goldens.csv
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
from typing import Any, Dict, List, Optional

import httpx
from gen_conversational_goldens import build_llm  # keep both files in the same folder

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
]


class ApiError(RuntimeError):
    def __init__(self, msg: str, retryable: bool):
        super().__init__(msg)
        self.retryable = retryable


# --------------------------------------------------------------------------- #
# SSE client for /answer  (event names/payloads taken from your controller.py)
# --------------------------------------------------------------------------- #
async def ask_once(
    url: str,
    headers: Dict[str, str],
    query: str,
    conversation_id: Optional[str],
    timeout_s: float,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"query": query}
    if conversation_id:
        body["conversation_id"] = conversation_id

    tokens: List[str] = []
    sources: List[dict] = []
    done: Optional[dict] = None
    timeout = httpx.Timeout(connect=10.0, read=timeout_s, write=30.0, pool=30.0)

    async with httpx.AsyncClient(timeout=timeout) as http:
        async with http.stream("POST", url, json=body, headers=headers) as r:
            if r.status_code != 200:
                detail = (await r.aread()).decode(errors="replace")[:300]
                raise ApiError(
                    f"/answer -> HTTP {r.status_code}: {detail}",
                    retryable=r.status_code >= 500,
                )
            event: Optional[str] = None
            data_lines: List[str] = []
            async for line in r.aiter_lines():
                if line == "":  # end of one SSE frame
                    if event is not None and data_lines:
                        data = json.loads("\n".join(data_lines))
                        if event == "token":
                            tokens.append(data.get("text", ""))
                        elif event == "reset":
                            tokens.clear()
                        elif event == "sources":
                            sources = data
                        elif event == "done":
                            done = data
                        elif event == "error":
                            raise ApiError(
                                f"server error event: {data.get('detail')}",
                                retryable=True,
                            )
                    event, data_lines = None, []
                elif line.startswith(":"):  # keepalive comment
                    continue
                elif line.startswith("event:"):
                    event = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:") :].lstrip())

    if done is None:
        raise ApiError(
            "stream ended without a 'done' event (nothing was saved server-side)",
            retryable=True,
        )
    return {
        "conversation_id": str(done["conversation_id"]),
        # `done.answer` is authoritative per your controller; fall back to streamed tokens.
        "answer": done.get("answer") or "".join(tokens),
        "sources": sources,
    }


async def ask_with_retry(url, headers, query, conversation_id, timeout_s, retries):
    for attempt in range(retries + 1):
        try:
            return await ask_once(url, headers, query, conversation_id, timeout_s)
        except ApiError as e:
            if not e.retryable or attempt == retries:
                raise
            wait = 2**attempt
            print(
                f"  [retry {attempt + 1}/{retries}] {e} - waiting {wait}s",
                file=sys.stderr,
            )
            await asyncio.sleep(wait)
        except (httpx.TransportError, json.JSONDecodeError) as e:
            if attempt == retries:
                raise
            wait = 2**attempt
            print(
                f"  [retry {attempt + 1}/{retries}] {type(e).__name__}: {e} - waiting {wait}s",
                file=sys.stderr,
            )
            await asyncio.sleep(wait)


# --------------------------------------------------------------------------- #
def parse_headers(items: List[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    env = os.environ.get("RAGWISE_AUTH_HEADER")
    for raw in ([env] if env else []) + list(items or []):
        if ":" not in raw:
            sys.exit(f"Bad header (expected 'Name: value'): {raw!r}")
        name, value = raw.split(":", 1)
        headers[name.strip()] = value.strip()
    return headers


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


def done_golden_ids(out_path: str) -> set:
    if not os.path.exists(out_path):
        return set()
    with open(out_path, newline="", encoding="utf-8") as f:
        return {r["golden_id"] for r in csv.DictReader(f)}


def run(args, llm=None) -> None:
    from deepeval.simulator import ConversationSimulator
    from deepeval.test_case import Turn

    headers = parse_headers(args.header)
    skip = done_golden_ids(args.out)
    goldens = load_goldens(args.goldens, skip, args.limit)
    if skip:
        print(f"Resuming: {len(skip)} conversations already in {args.out}")
    if not goldens:
        print("Nothing to simulate.")
        return
    print(
        f"Simulating {len(goldens)} conversations (max {args.max_user_turns} user turns each) against {args.api_url}"
    )

    # Side store: details of each assistant turn, keyed by the server's conversation id.
    # They are NOT put on the Turn, because DeepEval serialises every Turn field into the
    # simulated user's prompt (and the stop-check prompt) on later turns.
    conv_ids: Dict[str, str] = {}  # DeepEval thread_id -> server conversation_id
    details: Dict[str, List[dict]] = defaultdict(
        list
    )  # server conv id -> per-assistant-turn details

    async def model_callback(input: str, thread_id: str) -> Turn:
        t0 = time.perf_counter()
        res = await ask_with_retry(
            args.api_url,
            headers,
            input,
            conv_ids.get(thread_id),
            args.read_timeout,
            args.retries,
        )
        conv_ids[thread_id] = res["conversation_id"]
        details[res["conversation_id"]].append(
            {
                "chunk_ids": [s.get("chunk_id") for s in res["sources"]],
                "paper_ids": [s.get("paper_id") for s in res["sources"]],
                "contexts": [s.get("text") for s in res["sources"]],
                "latency_s": round(time.perf_counter() - t0, 2),
            }
        )
        return Turn(
            role="assistant",
            content=res["answer"],
            metadata={"ragwise_conversation_id": res["conversation_id"]},
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
            row = {
                "golden_id": gid,
                "turn_index": i,
                "role": turn.role,
                "content": turn.content,
            }
            if turn.role == "assistant":
                conv = (turn.metadata or {}).get("ragwise_conversation_id")
                d = details.get(conv, [])
                d = d[a_idx] if a_idx < len(d) else {}
                a_idx += 1
                row.update(
                    ragwise_conversation_id=conv,
                    retrieved_chunk_ids=json.dumps(d.get("chunk_ids", [])),
                    retrieved_paper_ids=json.dumps(d.get("paper_ids", [])),
                    retrieved_contexts=json.dumps(
                        d.get("contexts", []), ensure_ascii=False
                    ),
                    latency_s=d.get("latency_s"),
                )
            writer.writerow(row)
        out_f.flush()
        print(
            f"  saved {gid}: {sum(t.role == 'user' for t in test_case.turns)} user turns"
        )

    simulator = ConversationSimulator(
        model_callback=model_callback,
        simulator_model=llm or build_llm(args),
        max_concurrent=args.concurrency,
    )
    try:
        simulator.simulate(
            conversational_goldens=goldens,
            max_user_simulations=args.max_user_turns,
            on_simulation_complete=on_complete,
        )
    except Exception as e:
        print(
            f"\nSimulation aborted: {type(e).__name__}: {e}\n"
            f"Finished conversations are already in {args.out}; re-run the same command to resume.",
            file=sys.stderr,
        )
        raise
    finally:
        out_f.close()
    print(f"Done -> {args.out}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--goldens", default="conversational_goldens.csv")
    p.add_argument("--out", default="simulated_conversations.csv")
    p.add_argument("--api-url", default="http://localhost:8000/answer")
    p.add_argument(
        "--header",
        action="append",
        default=[],
        help="Repeatable 'Name: value'. Also read from env RAGWISE_AUTH_HEADER.",
    )
    p.add_argument("--max-user-turns", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument(
        "--limit", type=int, default=None, help="Only the first N not-yet-done goldens."
    )
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--read-timeout", type=float, default=300.0)
    p.add_argument(
        "--model", default=os.environ.get("OLLAMA_MODEL", "gemma4:31b-cloud")
    )
    p.add_argument(
        "--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Simulated-user temperature (a bit of variety makes users less robotic).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
