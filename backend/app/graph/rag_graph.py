from __future__ import annotations

import asyncio
import logging
from typing import Optional, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import QdrantClient, models

from app.services.rerank import CohereReranker
from app.services.retrieve_from_qdrant import (
    DEFAULT_COLLECTION,
    QueryEmbedder,
    retrieve_top_k,
)

logger = logging.getLogger(__name__)

# How many hybrid candidates to pull before reranking.
RERANK_CANDIDATES = 50

NO_DOCS_ANSWER = "I couldn't find any relevant documents to answer your question."

PROMPT = ChatPromptTemplate.from_template(
    """You are an expert AI assistant. Use the following pieces of retrieved context
to answer the user's question. If you don't know the answer based on the context,
just say you don't know. Don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""
)


class RetrievalError(Exception):
    """Raised when the retrieve node fails."""


class GenerationError(Exception):
    """Raised when the generate node fails."""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class _RAGInput(TypedDict):
    """Keys the caller must always provide when invoking the graph."""

    query: str
    top_k: int
    paper_id: Optional[str]
    has_picture: Optional[bool]


class RAGState(_RAGInput, total=False):
    candidates: list[dict]  # slim metadata of the 50 hybrid candidates (pre-rerank)
    chunks: list[dict]  # Cohere top_k, full chunks
    answer: str
    rerank_fallback: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_filter(
    paper_id: Optional[str], has_picture: Optional[bool]
) -> Optional[models.Filter]:
    must: list[models.Condition] = []
    if paper_id:
        must.append(
            models.FieldCondition(
                key="paper_id", match=models.MatchValue(value=paper_id)
            )
        )
    if has_picture is not None:
        must.append(
            models.FieldCondition(
                key="has_picture", match=models.MatchValue(value=has_picture)
            )
        )
    return models.Filter(must=must) if must else None


def _format_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"Paper: {c['paper_id']}\nContent: {c['text']}" for c in chunks
    )


def _slim_candidate(c: dict) -> dict:
    return {
        "rank": c["hybrid_rank"],
        "chunk_id": c["chunk_id"],
        "paper_id": c["paper_id"],
        "score": c["score"],  # RRF
    }


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------
def build_rag_graph(
    embedder: QueryEmbedder,
    client: QdrantClient,
    llm: ChatOllama,
    reranker: CohereReranker,
) -> CompiledStateGraph:
    chain = PROMPT | llm | StrOutputParser()

    async def retrieve(state: RAGState) -> dict:
        top_k = state["top_k"]
        candidate_k = max(RERANK_CANDIDATES, top_k)
        try:
            candidates = await asyncio.to_thread(
                retrieve_top_k,
                query=state["query"],
                client=client,
                embedder=embedder,
                collection=DEFAULT_COLLECTION,
                k=candidate_k,
                prefetch_limit=candidate_k,
                query_filter=_build_filter(
                    state.get("paper_id"), state.get("has_picture")
                ),
            )
        except Exception as e:
            raise RetrievalError(str(e)) from e
        if not candidates:
            return {"candidates": [], "chunks": []}
        # Tag each candidate with its hybrid rank before reranking. The
        # reranker copies the dict, so this survives into the final chunks
        # and you can see how far Cohere moved each one.
        for rank, c in enumerate(candidates, 1):
            c["hybrid_rank"] = rank
        rerank_fallback = False
        try:
            chunks = await reranker.rerank(
                query=state["query"], chunks=candidates, top_n=top_k
            )
        except Exception:
            logger.exception("Cohere rerank failed; falling back to hybrid order")
            chunks = candidates[:top_k]
            rerank_fallback = True

        return {
            "candidates": [_slim_candidate(c) for c in candidates],
            "chunks": chunks,
            "rerank_fallback": rerank_fallback,
        }

    async def generate(state: RAGState, config: RunnableConfig) -> dict:
        chunks = state.get("chunks", [])
        if not chunks:
            # Keeps the graph strictly linear: no LLM call when nothing was found.
            return {"answer": NO_DOCS_ANSWER}
        try:
            answer = await chain.ainvoke(
                {"context": _format_context(chunks), "question": state["query"]},
                config=config,
            )
        except Exception as e:
            raise GenerationError(str(e)) from e
        return {"answer": answer}

    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()
