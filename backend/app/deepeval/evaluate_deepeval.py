"""Simulate and evaluate multi-turn RAG conversations from a Langfuse dataset.

Each Langfuse dataset item is one conversational golden.  The item is simulated
against the graph used by the API, and the resulting conversation is evaluated
with DeepEval's aggregate ``TurnContextualRelevancyMetric``.  Langfuse's
``run_experiment`` stores the conversation output and the score on the dataset
run item while the graph's LangChain observations remain nested under the item
trace.

Run from ``backend/``::

    uv run python -m app.deepeval.evaluate_deepeval --limit 1
    uv run python -m app.deepeval.evaluate_deepeval --run-name baseline

The source CSV files used to create the dataset are intentionally not read or
written here.  They remain useful as local inspection artifacts, but Langfuse
is the source of truth for this workflow.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, cast

# Keep ``--help`` usable without importing the application and its service
# clients.  These paths also support invoking this file directly from either
# ``backend/`` or ``backend/app/deepeval/``.
_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent.parent
for _path in (_HERE, _BACKEND):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


log = logging.getLogger("evaluate_deepeval")

DEFAULT_DATASET = "ragwise_deepeval_conversational_goldens"
EXPERIMENT_NAME = "ragwise-deepeval-conversational-eval"
DEFAULT_TOP_K = 5
DEFAULT_MAX_USER_TURNS = 3
DEFAULT_RERANK_INTERVAL_S = 6.1
CONTEXTUAL_RELEVANCY_NAME = "contextual_relevancy"


def _field(item: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a Langfuse DatasetItem or a mapping."""

    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _as_list(value: Any) -> list[Any]:
    """Decode a dataset metadata list stored either as JSON or as a list."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            return []
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return [value]
        if decoded is None:
            return []
        return decoded if isinstance(decoded, list) else [decoded]
    return [value]


def _item_metadata(item: Any) -> dict[str, Any]:
    metadata = _field(item, "metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def item_id(item: Any) -> str:
    """Return the stable golden identifier used for the Langfuse session."""

    metadata = _item_metadata(item)
    golden_id = metadata.get("golden_id")
    if golden_id:
        return str(golden_id)

    dataset_item_id = _field(item, "id")
    if dataset_item_id:
        return str(dataset_item_id)

    # Langfuse dataset items normally have an id.  This fallback keeps local
    # tests deterministic when they use a small mapping instead.
    digest = hashlib.sha256(str(_field(item, "input", "")).encode()).hexdigest()
    return digest[:16]


class _RerankRateLimiter:
    """Reserve globally spaced reranker starts across experiment tasks.

    Langfuse may execute task coroutines concurrently, while DeepEval runs its
    simulator in a worker thread to avoid nesting event loops.  A threading
    lock therefore keeps the Cohere pacing correct across both boundaries.
    """

    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self._lock = threading.Lock()
        self._next_start = 0.0

    async def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_start)
            self._next_start = start + self.interval_s
        delay = start - now
        if delay > 0:
            await asyncio.sleep(delay)


def build_graph(args: argparse.Namespace):
    """Build the same RAG graph and clients used by the application."""

    from qdrant_client import QdrantClient

    from app.config.config import settings
    from app.graph.rag_graph import build_rag_graph
    from app.services.llm_factory import build_llms
    from app.services.rerank import CohereReranker
    from app.services.retrieve_from_qdrant import QueryEmbedder

    log.info(
        "Loading BGE-M3 + Qdrant + generator (%s via %s, images=%s)",
        settings.llm_model,
        settings.llm_base_url,
        "on" if args.use_images else "off",
    )
    embedder = QueryEmbedder()
    embedder.load()

    qdrant = QdrantClient(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY"),
        prefer_grpc=True,
        grpc_port=6334,
    )
    llms = build_llms()
    reranker = CohereReranker(
        api_key=settings.cohere_api_key,
        model=settings.cohere_rerank_model,
    )
    graph = build_rag_graph(
        embedder=embedder,
        client=qdrant,
        llm=llms.main,
        reranker=reranker,
        use_images=args.use_images,
        rewrite_llm=llms.rewrite,
        use_cache=args.use_cache,
        cache_options=settings.cache_options(),
    )
    return graph, qdrant


def _strip_fences(text: str) -> str:
    """Remove markdown JSON fences emitted by some Ollama judge models."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()


def build_judge(args: argparse.Namespace):
    """Build a fence-tolerant DeepEval Ollama judge."""

    from deepeval.models import OllamaModel
    from deepeval.models.llms.ollama_model import retry_ollama
    from deepeval.test_case import Audio
    from pydantic import BaseModel

    class FenceTolerantOllama(OllamaModel):
        """Allow structured judge responses wrapped in markdown fences."""

        def _parse(self, content: str, schema: Optional[BaseModel]):
            if not schema:
                return content
            cleaned = _strip_fences(content)
            try:
                return schema.model_validate_json(cleaned)
            except Exception:
                # DeepEval can perform a second JSON extraction pass when the
                # raw cleaned string is returned instead of raising here.
                return cleaned

        @retry_ollama
        def generate(
            self,
            prompt: str,
            schema: Optional[BaseModel] = None,
            audio: Optional[list[Audio]] = None,
        ) -> tuple[Any, float]:
            client = self.load_model()
            response = client.chat(
                model=self.name,
                messages=self._prompt_messages(prompt, audio),
                format=schema.model_json_schema() if schema else None,
                options={"temperature": self.temperature, **self.generation_kwargs},
            )
            return self._parse(response.message.content, schema), 0

        @retry_ollama
        async def a_generate(
            self,
            prompt: str,
            schema: Optional[BaseModel] = None,
            audio: Optional[list[Audio]] = None,
        ) -> tuple[Any, float]:
            client = self.load_model(async_mode=True)
            response = await client.chat(
                model=self.name,
                messages=self._prompt_messages(prompt, audio),
                format=schema.model_json_schema() if schema else None,
                options={"temperature": self.temperature, **self.generation_kwargs},
            )
            return self._parse(response.message.content, schema), 0

    kwargs: dict[str, Any] = {}
    api_key = os.environ.get("OLLAMA_API_KEY")
    if api_key:
        kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}
    return FenceTolerantOllama(
        model=args.judge_model,
        base_url=args.judge_host,
        temperature=args.judge_temperature,
        **kwargs,
    )


def _make_golden(item: Any):
    from deepeval.dataset import ConversationalGolden

    metadata = _item_metadata(item)
    metadata.setdefault("golden_id", item_id(item))
    return ConversationalGolden(
        scenario=str(_field(item, "input", "")),
        expected_outcome=_field(item, "expected_output") or None,
        context=[str(value) for value in _as_list(metadata.get("context_chunks"))],
        name=item_id(item),
        user_description=None,
        multimodal=False,
        additional_metadata=metadata,
    )


def _serialize_test_case(
    test_case: Any,
    details: dict[str, list[dict[str, Any]]],
    session_id: str,
) -> list[dict[str, Any]]:
    """Convert DeepEval turns plus graph details into Langfuse-safe data."""

    serialized: list[dict[str, Any]] = []
    assistant_index = 0
    for turn in test_case.turns:
        row: dict[str, Any] = {
            "role": turn.role,
            "content": turn.content,
        }
        if turn.role == "assistant":
            turn_metadata = turn.metadata or {}
            thread_id = turn_metadata.get("thread_id")
            thread_details = (
                details.get(thread_id, []) if isinstance(thread_id, str) else []
            )
            detail = (
                thread_details[assistant_index]
                if assistant_index < len(thread_details)
                else {}
            )
            assistant_index += 1
            row.update(
                {
                    "conversation_id": session_id,
                    "retrieval_context": list(detail.get("contexts", [])),
                    "chunk_ids": list(detail.get("chunk_ids", [])),
                    "paper_ids": list(detail.get("paper_ids", [])),
                    "rewritten_query": detail.get("rewritten_query"),
                    "rerank_fallback": bool(detail.get("rerank_fallback", False)),
                    "images_used": list(detail.get("images_used", [])),
                    "history_len": detail.get("history_len"),
                }
            )
        serialized.append(row)
    return serialized


def make_task(
    graph: Any,
    args: argparse.Namespace,
    simulated_user_model: Any,
    run_name: str,
) -> Any:
    """Create the Langfuse task that simulates one dataset item."""

    from deepeval.simulator import ConversationSimulator
    from deepeval.test_case import Turn

    # Imported here so app imports remain deferred for --help.
    from langfuse import propagate_attributes
    from langfuse.langchain import CallbackHandler

    from app.graph.rag_graph import GenerationError, RetrievalError

    rerank_limiter = _RerankRateLimiter(DEFAULT_RERANK_INTERVAL_S)

    async def task(*, item: Any, **kwargs: Any) -> dict[str, Any]:
        gid = item_id(item)
        session_id = f"{run_name}:{gid}"
        details: dict[str, list[dict[str, Any]]] = defaultdict(list)
        histories: dict[str, list[dict[str, str]]] = defaultdict(list)
        failures: list[str] = []
        golden: Any = None

        async def model_callback(*, input: str, thread_id: str) -> Turn:
            user_input = input
            await rerank_limiter.wait()
            history = histories[thread_id]
            history_len = len(history)
            state: dict[str, Any] | None = None
            last_error: Optional[Exception] = None

            for attempt in range(args.retries + 1):
                input_state = {
                    "query": user_input,
                    "top_k": args.top_k,
                    "paper_id": None,
                    "has_picture": None,
                    "history": list(history),
                    "history_summary": None,
                }
                try:
                    if args.trace:
                        with propagate_attributes(
                            trace_name="deepeval-conversation",
                            user_id=args.eval_user_id,
                            session_id=session_id,
                            tags=["rag", "eval-sim", "deepeval"],
                            metadata={
                                "golden_id": gid,
                                "dataset": args.dataset,
                                "run_name": run_name,
                                "turn": str(history_len // 2 + 1),
                                "top_k": str(args.top_k),
                                "use_images": str(args.use_images).lower(),
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
                except (RetrievalError, GenerationError) as exc:
                    last_error = exc
                    if attempt < args.retries:
                        delay = 2**attempt
                        log.warning(
                            "Graph retry %d/%d for %s: %s; sleeping %ss",
                            attempt + 1,
                            args.retries,
                            gid,
                            exc,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    break
                except Exception as exc:  # keep one bad item from aborting the run
                    last_error = exc
                    break

            if state is None:
                error_text = f"{type(last_error).__name__}: {last_error}"
                failures.append(error_text)
                log.error("Graph failed for %s: %s", gid, error_text)
                state = {
                    "answer": "",
                    "chunks": [],
                    "rewritten_query": user_input,
                    "rerank_fallback": True,
                    "images_used": [],
                }

            answer = str(state.get("answer") or "")
            chunks = state.get("chunks") or []
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": answer})
            details[thread_id].append(
                {
                    "chunk_ids": [chunk.get("chunk_id") for chunk in chunks],
                    "paper_ids": [chunk.get("paper_id") for chunk in chunks],
                    "contexts": [chunk.get("text", "") for chunk in chunks],
                    "rewritten_query": state.get("rewritten_query"),
                    "rerank_fallback": bool(state.get("rerank_fallback", False)),
                    "images_used": [
                        {
                            "n": image.get("n"),
                            "chunk_id": image.get("chunk_id"),
                            "paper_id": image.get("paper_id"),
                            "path": image.get("path"),
                        }
                        for image in (state.get("images_used") or [])
                    ],
                    "history_len": history_len,
                }
            )
            return Turn(
                role="assistant",
                content=answer,
                metadata={
                    "conversation_id": session_id,
                    "thread_id": thread_id,
                },
            )

        async def simulate() -> list[dict[str, Any]]:
            simulator = ConversationSimulator(
                model_callback=cast(Any, model_callback),
                simulator_model=simulated_user_model,
                # Langfuse controls item-level concurrency.  There is only one
                # golden in this simulator, so nested concurrency is unnecessary.
                max_concurrent=1,
            )
            # DeepEval's public simulate() is synchronous and nests an event
            # loop. Langfuse already runs this task asynchronously, so use the
            # async single-conversation implementation directly.
            simulate_one = getattr(simulator, "_a_simulate_single_conversation", None)
            if simulate_one is None:
                raise RuntimeError(
                    "Installed DeepEval does not expose its async conversation simulator"
                )
            # DeepEval's run wrapper initializes this before the first model call.
            simulator.simulation_cost = 0 if simulator.using_native_model else None
            test_case = await simulate_one(
                golden=golden,
                max_user_simulations=args.max_user_turns,
                index=0,
            )
            return _serialize_test_case(test_case, details, session_id)

        try:
            golden = _make_golden(item)
            # Keep this context around the complete simulation. Langfuse's
            # experiment item trace becomes the parent of every graph callback,
            # while session_id groups all turns as one conversation.
            with propagate_attributes(
                trace_name="deepeval-conversation",
                user_id=args.eval_user_id,
                session_id=session_id,
                tags=["rag", "eval-sim", "deepeval"],
                metadata={
                    "golden_id": gid,
                    "dataset": args.dataset,
                    "run_name": run_name,
                    "max_user_turns": str(args.max_user_turns),
                },
            ):
                turns = await simulate()
            status = "failed" if failures else "success"
            return {
                "conversation_id": session_id,
                "golden_id": gid,
                "status": status,
                "error": "; ".join(failures)[:1000] if failures else None,
                "scenario": str(_field(item, "input", "")),
                "turns": turns,
            }
        except (
            Exception
        ) as exc:  # task errors are represented and evaluated as failures
            error_text = f"{type(exc).__name__}: {exc}"
            log.exception("Conversation simulation failed for %s", gid)
            return {
                "conversation_id": session_id,
                "golden_id": gid,
                "status": "failed",
                "error": error_text[:1000],
                "scenario": str(_field(item, "input", "")),
                "turns": [],
            }

    return task


def _case_from_output(output: dict[str, Any], input_text: str, expected_output: Any):
    from deepeval.test_case import ConversationalTestCase, Turn

    turns = []
    for serialized in output.get("turns", []):
        retrieval_context = serialized.get("retrieval_context")
        turns.append(
            Turn(
                role=serialized["role"],
                content=str(serialized.get("content", "")),
                retrieval_context=(
                    [str(value) for value in retrieval_context]
                    if retrieval_context
                    else None
                ),
            )
        )
    return ConversationalTestCase(
        turns=turns,
        scenario=input_text,
        expected_outcome=expected_output or None,
        name=output.get("golden_id"),
    )


def make_evaluator(args: argparse.Namespace, judge: Any) -> Any:
    """Create the Langfuse evaluator for one aggregate conversation score."""

    async def contextual_relevancy_evaluator(
        *,
        input: Any,
        output: Any,
        expected_output: Any = None,
        **kwargs: Any,
    ) -> Any:
        from deepeval.metrics import TurnContextualRelevancyMetric
        from langfuse import Evaluation

        if not isinstance(output, dict):
            return Evaluation(
                name=CONTEXTUAL_RELEVANCY_NAME,
                value=0.0,
                comment="simulation returned no structured output",
                metadata={"status": "failed"},
            )

        if output.get("status") != "success":
            return Evaluation(
                name=CONTEXTUAL_RELEVANCY_NAME,
                value=0.0,
                comment=f"simulation failed: {output.get('error') or 'unknown error'}",
                metadata={
                    "status": "simulation_failed",
                    "error": str(output.get("error") or "unknown error")[:1000],
                },
            )

        try:
            case = _case_from_output(output, str(input), expected_output)
            if not any(turn.role == "assistant" for turn in case.turns):
                raise ValueError("conversation contains no assistant turns")

            # Instantiate per item because DeepEval stores score/reason on the
            # metric object and Langfuse can evaluate items concurrently.
            metric = TurnContextualRelevancyMetric(
                model=judge,
                include_reason=True,
                async_mode=True,
            )
            score = float(await metric.a_measure(case, _show_indicator=False))
            if math.isnan(score) or math.isinf(score):
                raise ValueError("metric returned a non-finite score")
            return Evaluation(
                name=CONTEXTUAL_RELEVANCY_NAME,
                value=score,
                comment=metric.reason or "",
                metadata={
                    "status": "success",
                    "judge_model": args.judge_model,
                },
            )
        except Exception as exc:  # preserve the item and continue the experiment
            error_text = f"{type(exc).__name__}: {exc}"
            log.exception("Contextual relevancy evaluation failed: %s", error_text)
            return Evaluation(
                name=CONTEXTUAL_RELEVANCY_NAME,
                value=0.0,
                comment=f"evaluation failed: {error_text}",
                metadata={
                    "status": "evaluation_failed",
                    "error": error_text[:1000],
                },
            )

    return contextual_relevancy_evaluator


def _average_scores(result: Any) -> tuple[float, int, int]:
    values: list[float] = []
    total = len(result.item_results)
    for item_result in result.item_results:
        for evaluation in item_result.evaluations:
            if (
                evaluation.name == CONTEXTUAL_RELEVANCY_NAME
                and isinstance(evaluation.value, (int, float))
                and math.isfinite(float(evaluation.value))
            ):
                values.append(float(evaluation.value))
    return (sum(values) / len(values) if values else float("nan"), len(values), total)


def run(args: argparse.Namespace) -> Any:
    from gen_conversational_goldens import build_llm
    from langfuse import Langfuse

    from app.config.config import settings

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_name = args.run_name or (
        f"{EXPERIMENT_NAME}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )
    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
    )

    graph, qdrant = build_graph(args)
    try:
        dataset = langfuse.get_dataset(args.dataset)
        items = list(dataset.items)
        if args.limit is not None:
            items = items[: args.limit]
        if not items:
            raise ValueError(f"Langfuse dataset '{args.dataset}' contains no items")

        log.info("Dataset '%s': evaluating %d conversations", args.dataset, len(items))
        simulated_user_model = build_llm(args)
        judge = build_judge(args)
        result = langfuse.run_experiment(
            name=EXPERIMENT_NAME,
            run_name=run_name,
            description=(
                "DeepEval conversation simulation with aggregate "
                "contextual relevancy evaluation"
            ),
            data=items,
            task=make_task(graph, args, simulated_user_model, run_name),
            evaluators=[make_evaluator(args, judge)],
            max_concurrency=args.concurrency,
            metadata={
                "dataset": args.dataset,
                "top_k": str(args.top_k),
                "max_user_turns": str(args.max_user_turns),
                "simulator_model": args.model,
                "judge_model": args.judge_model,
                "judge_host": args.judge_host,
                "generator_model": settings.llm_model,
                "generator_base_url": settings.llm_base_url,
                "use_images": str(args.use_images),
                "use_cache": str(args.use_cache),
                "metric": CONTEXTUAL_RELEVANCY_NAME,
            },
        )
        average, scored, total = _average_scores(result)
        print(f"\n=== DeepEval summary (run: {result.run_name}) ===")
        print(f"{CONTEXTUAL_RELEVANCY_NAME:24s} {average:.3f} (n={scored}/{total})")
        if result.dataset_run_url:
            print(f"\nDataset run: {result.dataset_run_url}")
        return result
    finally:
        langfuse.flush()
        qdrant.close()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Langfuse conversational dataset name.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Langfuse dataset-run name (default: timestamped).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N dataset items.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent Langfuse dataset items; keep low for Ollama/Cohere.",
    )

    simulation = parser.add_argument_group("Simulation")
    simulation.add_argument(
        "--max-user-turns", type=int, default=DEFAULT_MAX_USER_TURNS
    )
    simulation.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Graph retries per turn for retrieval/generation failures.",
    )
    simulation.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "gemma4:31b-cloud"),
        help="DeepEval simulated-user Ollama model.",
    )
    simulation.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama host used by the simulated user.",
    )
    simulation.add_argument("--temperature", type=float, default=0.7)

    graph = parser.add_argument_group("RAG graph")
    graph.add_argument(
        "--use-cache",
        action="store_true",
        help="Opt into semantic caching; disabled for baseline evaluations",
    )
    graph.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    graph.add_argument(
        "--no-images",
        dest="use_images",
        action="store_false",
        help="Use text-only generation.",
    )
    graph.set_defaults(use_images=True)

    judge = parser.add_argument_group("DeepEval judge")
    judge.add_argument(
        "--judge-model",
        default=os.environ.get(
            "JUDGE_MODEL",
            os.environ.get("OLLAMA_JUDGE_MODEL", "gemma4:31b-cloud"),
        ),
    )
    judge.add_argument(
        "--judge-host",
        default=os.environ.get(
            "JUDGE_OLLAMA_HOST",
            os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        ),
    )
    judge.add_argument("--judge-temperature", type=float, default=0.0)

    langfuse = parser.add_argument_group("Langfuse")
    langfuse.add_argument(
        "--eval-user-id",
        default="eval-sim",
        help="Langfuse user_id attached to simulated conversation traces.",
    )
    langfuse.add_argument(
        "--no-trace",
        dest="trace",
        action="store_false",
        default=True,
        help="Disable graph-level Langfuse callbacks (experiment results remain enabled).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
