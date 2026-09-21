"""
Evaluate the RAG graph on a Langfuse dataset (testset) with RAGAS,
using a local Ollama model as both generator and judge, and store
the scores in Langfuse.

Built on `Langfuse.run_experiment` (Langfuse Python SDK v4). For every
dataset item it:
    1. runs the RAG graph (retrieve + generate) -> answer + retrieved chunks
    2. scores the sample with 4 RAGAS metrics
    3. attaches the 4 scores to that item's trace / dataset run item

Usage:
    python -m app.ragas.evaluate_rag --limit 1           # smoke test
    python -m app.ragas.evaluate_rag --run-name v1 --top-k 5

Requirements:
    pip install ragas langchain-ollama langchain-huggingface sentence-transformers \
                langfuse langgraph qdrant-client httpx
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from typing import Any

import httpx
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langfuse import Evaluation, Langfuse
from langfuse.langchain import CallbackHandler
from qdrant_client import QdrantClient
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from app.config import settings
from app.graph.rag_graph import build_rag_graph
from app.services.retrieve_from_qdrant import QueryEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("evaluate_rag")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_JUDGE_MODEL = os.environ.get("OLLAMA_JUDGE_MODEL", "gemma4:31b-cloud")
OLLAMA_GENERATOR_MODEL = os.environ.get("OLLAMA_GENERATOR_MODEL", "gemma4:31b-cloud")
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))

# ResponseRelevancy needs an embedding model: BGE-M3 from the local HF cache.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")


# ---------------------------------------------------------------------------
# Health check — fail fast if Ollama is unreachable or models are missing
# ---------------------------------------------------------------------------
def check_ollama(base_url: str, judge_model: str, generator_model: str) -> None:
    """Exit with a clear error if Ollama is down or a model is missing."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except httpx.ConnectError:
        log.error(
            "Ollama is not reachable at %s — "
            "run `ollama serve` in another terminal, then retry.",
            base_url,
        )
        sys.exit(1)
    except Exception as exc:
        log.error("Error connecting to Ollama at %s: %s", base_url, exc)
        sys.exit(1)

    available: set[str] = set()
    for m in resp.json().get("models", []):
        if name := m.get("name"):
            available.add(name)

    for label, model in (("judge", judge_model), ("generator", generator_model)):
        if model not in available:
            log.error(
                "Ollama model '%s' (%s) not found. "
                "Pull it with:  ollama pull %s\n"
                "Available models: %s",
                model,
                label,
                model,
                ", ".join(sorted(available)) or "(none)",
            )
            sys.exit(1)

    log.info(
        "Ollama OK  —  judge=%s  generator=%s  (at %s)",
        judge_model,
        generator_model,
        base_url,
    )


# ---------------------------------------------------------------------------
# Dataset item access
# ---------------------------------------------------------------------------
def _field(item: Any, name: str) -> Any:
    return item[name] if isinstance(item, dict) else getattr(item, name)


# ---------------------------------------------------------------------------
# RAGAS setup
# ---------------------------------------------------------------------------
def build_metrics() -> dict[str, Any]:
    judge = ChatOllama(
        model=OLLAMA_JUDGE_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=OLLAMA_NUM_CTX,
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    ragas_llm = LangchainLLMWrapper(judge)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)

    return {
        "context_precision": LLMContextPrecisionWithReference(llm=ragas_llm),
        "context_recall": LLMContextRecall(llm=ragas_llm),
        "faithfulness": Faithfulness(llm=ragas_llm),
        "answer_relevancy": ResponseRelevancy(llm=ragas_llm, embeddings=ragas_emb),
    }


# ---------------------------------------------------------------------------
# Experiment task + evaluator
# ---------------------------------------------------------------------------
def make_task(graph, top_k: int):
    async def task(*, item, **kwargs) -> dict:
        """Run the RAG graph. Output carries the contexts so the evaluator can use them."""
        question = _field(item, "input")
        handler = CallbackHandler()
        result = await graph.ainvoke(
            {
                "query": question,
                "top_k": top_k,
                "paper_id": None,
                "has_picture": None,
            },
            config={"callbacks": [handler]},
        )
        chunks = result.get("chunks", [])
        return {
            "answer": result["answer"],
            "contexts": [c["text"] for c in chunks],
            "paper_ids": [c["paper_id"] for c in chunks],
        }

    return task


def make_evaluator(metrics: dict[str, Any]):
    async def ragas_evaluator(
        *, input, output, expected_output, metadata=None, **kwargs
    ) -> list[Evaluation]:
        if not isinstance(expected_output, str) or not expected_output.strip():
            log.warning("skipping item without a reference answer: %.80s", input)
            return []

        contexts: list[str] = output["contexts"]
        if not contexts:
            return [
                Evaluation(
                    name="context_precision", value=0.0, comment="no chunks retrieved"
                ),
                Evaluation(
                    name="context_recall", value=0.0, comment="no chunks retrieved"
                ),
            ]

        sample = SingleTurnSample(
            user_input=input,
            retrieved_contexts=contexts,
            response=output["answer"],
            reference=expected_output,
        )

        evaluations: list[Evaluation] = []
        for name, metric in metrics.items():
            try:
                value = float(await metric.single_turn_ascore(sample))
            except Exception as e:  # noqa: BLE001
                log.warning("metric %s failed on %.60r: %s", name, input, e)
                continue
            if math.isnan(value):
                log.warning("metric %s returned NaN on %.60r", name, input)
                continue
            evaluations.append(
                Evaluation(
                    name=name,
                    value=value,
                    comment=f"ragas, judge={OLLAMA_JUDGE_MODEL}",
                )
            )
        return evaluations

    return ragas_evaluator


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    # --- Fail fast if Ollama is down or models are missing ----------------
    check_ollama(OLLAMA_BASE_URL, OLLAMA_JUDGE_MODEL, OLLAMA_GENERATOR_MODEL)

    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
    )

    # --- RAG graph (generator now uses Ollama, not Gemini) -----------------
    log.info(
        "Loading BGE-M3 + Qdrant + generator (Ollama: %s) ...",
        OLLAMA_GENERATOR_MODEL,
    )
    embedder = QueryEmbedder()
    embedder.load()
    qdrant = QdrantClient(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY"),
        prefer_grpc=True,
        grpc_port=6334,
    )
    generator_llm = ChatOllama(
        model=OLLAMA_GENERATOR_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
        num_ctx=OLLAMA_NUM_CTX,
    )
    graph = build_rag_graph(embedder=embedder, client=qdrant, llm=generator_llm)

    metrics = build_metrics()

    # --- Dataset ------------------------------------------------------------
    dataset = langfuse.get_dataset(args.dataset)
    items = list(dataset.items)
    if args.limit:
        items = items[: args.limit]
    log.info("Dataset '%s': evaluating %d items", args.dataset, len(items))

    try:
        result = langfuse.run_experiment(
            name="ragwise-ragas-eval",
            run_name=args.run_name,
            description="RAGAS: context precision/recall, faithfulness, answer relevancy",
            data=items,
            task=make_task(graph, args.top_k),
            evaluators=[make_evaluator(metrics)],
            max_concurrency=args.concurrency,
            metadata={
                "top_k": str(args.top_k),
                "judge_model": OLLAMA_JUDGE_MODEL,
                "generator_model": OLLAMA_GENERATOR_MODEL,
                "embed_model": EMBED_MODEL,
            },
        )

        # --- Summary --------------------------------------------------------
        totals: dict[str, list[float]] = {}
        for r in result.item_results:
            for ev in r.evaluations:
                if isinstance(ev.value, (int, float)):
                    totals.setdefault(ev.name, []).append(float(ev.value))

        print(f"\n=== Averages (run: {result.run_name}) ===")
        for name in (
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
        ):
            vals = totals.get(name, [])
            avg = sum(vals) / len(vals) if vals else float("nan")
            print(f"{name:20s} {avg:.3f}   (n={len(vals)}/{len(items)})")
        url = getattr(result, "dataset_run_url", None)
        if url:
            print(f"\nDataset run: {url}")
    finally:
        langfuse.flush()
        qdrant.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dataset", default="ragwise_ragas_testset", help="Langfuse dataset name"
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="Dataset run name (default: auto with timestamp)",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--limit", type=int, default=None, help="Only evaluate the first N items"
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Items processed in parallel. Keep low for a single local GPU.",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
