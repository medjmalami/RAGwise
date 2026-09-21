from __future__ import annotations

import asyncio
from typing import Optional, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import QdrantClient, models

from app.services.retrieve_from_qdrant import (
    DEFAULT_COLLECTION,
    QueryEmbedder,
    retrieve_top_k,
)

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
    """Full graph state: required inputs + keys written by the nodes."""

    chunks: list[dict]  # written by `retrieve`
    answer: str  # written by `generate`


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


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------
def build_rag_graph(
    embedder: QueryEmbedder,
    client: QdrantClient,
    llm: ChatOllama,
) -> CompiledStateGraph:
    chain = PROMPT | llm | StrOutputParser()

    async def retrieve(state: RAGState) -> dict:
        try:
            chunks = await asyncio.to_thread(
                retrieve_top_k,
                query=state["query"],
                client=client,
                embedder=embedder,
                collection=DEFAULT_COLLECTION,
                k=state["top_k"],
                query_filter=_build_filter(
                    state.get("paper_id"), state.get("has_picture")
                ),
            )
        except Exception as e:
            raise RetrievalError(str(e)) from e
        return {"chunks": chunks}

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
