from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Optional, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langfuse import propagate_attributes
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import QdrantClient, models

from app.services.rerank import CohereReranker
from app.services.retrieve_from_qdrant import (
    DEFAULT_COLLECTION,
    QueryEmbedder,
    retrieve_top_k,
)
from app.services.vision_llm import PathImageChatOllama, image_path_block

logger = logging.getLogger(__name__)

# How many hybrid candidates to pull before reranking.
RERANK_CANDIDATES = 50

# Image limits for the generate step (tune after looking at latency/quality).
MAX_IMAGES_PER_CHUNK = 3
MAX_IMAGES_TOTAL = 6

NO_DOCS_ANSWER = "I couldn't find any relevant documents to answer your question."

# `{image_note}` is "" when no image is attached, so the text-only prompt stays
# byte-identical to the old one (keeps your baseline comparable).
SYSTEM_PROMPT = """You are an expert AI assistant. Use the retrieved context provided \
with the latest question to answer it. If you don't know the answer based on the \
context, just say you don't know. Don't try to make up an answer."""

# Per-turn: context and question. `{image_note}` is "" when no image is
# attached, otherwise IMAGE_NOTE (which ends with a blank line).
HUMAN_PROMPT = ChatPromptTemplate.from_template(
    """{image_note}Context:
{context}

Question: {question}

Answer:"""
)

IMAGE_NOTE = (
    "Some context chunks come with attached images, referenced as 'Image N'. "
    "Use them together with the chunk text when they are relevant to the question."
    "\n\n"
)

# Rewrite step: how much history the rewriter sees, and sanity limits.
REWRITE_MAX_TURNS = 3  # last N user/assistant exchanges
REWRITE_MAX_CHARS_PER_MSG = 600
REWRITE_MAX_QUERY_CHARS = 400

REWRITE_SYSTEM_PROMPT = """You rewrite follow-up questions into standalone search queries \
for a retrieval system over scientific papers (arXiv).

Given the conversation so far and the user's latest message, produce ONE self-contained \
query that can be understood without the conversation.

Rules:
- Resolve pronouns and references ("it", "that method", "the second one", "what about X?") \
using the conversation.
- Keep technical terms, model/dataset names, paper IDs and numbers exactly as written.
- If the latest message is already standalone, return it unchanged.
- Do NOT answer the question. Do NOT add information that is not in the conversation.
- Reply in the same language as the latest message.
- Output only the query: no quotes, no preamble, no explanation."""

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", REWRITE_SYSTEM_PROMPT),
        (
            "human",
            "Conversation:\n{history}\n\nLatest message: {question}\n\nStandalone query:",
        ),
    ]
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
    history: list[dict]  # [{"role": "user"|"assistant", "content": str}], oldest first


class RAGState(_RAGInput, total=False):
    rewritten_query: str  # standalone query used for retrieval + rerank
    candidates: list[dict]  # slim metadata of the 50 hybrid candidates (pre-rerank)
    chunks: list[dict]  # Cohere top_k, full chunks (text untouched by the image step)
    answer: str
    rerank_fallback: bool
    images_used: list[dict]  # [{n, chunk_id, paper_id, path}] -- paths only, no bytes
    images_skipped: list[dict]  # [{chunk_id, path, reason}]
    image_fallback: bool  # True if the LLM call with images failed -> text-only retry


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


def _format_context(
    chunks: list[dict], images_by_chunk: Optional[dict[str, list[int]]] = None
) -> str:
    parts = []
    for c in chunks:
        part = f"Paper: {c['paper_id']}\nContent: {c['text']}"
        nums = (images_by_chunk or {}).get(c.get("chunk_id") or "")
        if nums:
            # Ollama attaches images to the message, not at a position in the
            # text, so this label is what ties each image to its chunk.
            part += "\nAttached for this chunk: " + ", ".join(
                f"Image {n}" for n in nums
            )
        parts.append(part)
    return "\n\n---\n\n".join(parts)


def _slim_candidate(c: dict) -> dict:
    return {
        "rank": c["hybrid_rank"],
        "chunk_id": c["chunk_id"],
        "paper_id": c["paper_id"],
        "score": c["score"],  # RRF
    }


def _collect_images(chunks: list[dict], root: Path) -> tuple[list[dict], list[dict]]:
    """Pick the images to send, in chunk order. Returns (used, skipped)."""
    used: list[dict] = []
    skipped: list[dict] = []
    for c in chunks:
        payload = c.get("payload") or {}
        if not payload.get("has_picture"):
            continue
        paths = payload.get("image_paths") or []
        if not paths:
            skipped.append(
                {
                    "chunk_id": c.get("chunk_id"),
                    "path": None,
                    "reason": "has_picture but no image_paths in payload",
                }
            )
            continue
        for i, rel in enumerate(paths):
            meta = {"chunk_id": c.get("chunk_id"), "path": rel}
            if i >= MAX_IMAGES_PER_CHUNK or len(used) >= MAX_IMAGES_TOTAL:
                skipped.append({**meta, "reason": "cap"})
            elif not (root / rel).is_file():
                skipped.append({**meta, "reason": "missing_file"})
            else:
                used.append(
                    {
                        "n": len(used) + 1,
                        "chunk_id": c.get("chunk_id"),
                        "paper_id": c.get("paper_id"),
                        "path": rel,
                    }
                )
    return used, skipped


def _trace_config(
    config: RunnableConfig,
    used: list[dict],
    skipped: list[dict],
    fallback: bool = False,
) -> RunnableConfig:
    tags = ["with_images" if used else "text_only"]

    if skipped:
        tags.append("images_skipped")
    if fallback:
        tags.append("image_fallback")

    existing_metadata = config.get("metadata") or {}
    existing_tags = existing_metadata.get("langfuse_tags") or []

    meta = {
        "images_used": bool(used),
        "num_images": len(used),
        "image_status": "used" if used else "text_only",
        "image_paths": [u["path"] for u in used],
        "image_chunk_ids": [u["chunk_id"] for u in used],
        "image_paper_ids": sorted({u["paper_id"] for u in used}),
        "images_skipped": skipped,
        "langfuse_tags": [*existing_tags, *tags],
    }

    return cast(
        RunnableConfig,
        {
            **config,
            "tags": [*(config.get("tags") or []), *tags],
            "metadata": {**existing_metadata, **meta},
        },
    )


def _history_messages(history: list[dict]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for m in history:
        if m.get("role") == "user":
            messages.append(HumanMessage(content=m["content"]))
        elif m.get("role") == "assistant":
            messages.append(AIMessage(content=m["content"]))
    return messages


def _build_messages(
    chunks: list[dict],
    query: str,
    used: list[dict],
    history: Optional[list[dict]] = None,
) -> list[BaseMessage]:
    images_by_chunk: dict[str, list[int]] = {}
    for u in used:
        images_by_chunk.setdefault(u["chunk_id"] or "", []).append(u["n"])
    msg = HUMAN_PROMPT.format_messages(
        context=_format_context(chunks, images_by_chunk),
        question=query,
        image_note=IMAGE_NOTE if used else "",
    )[0]

    final: BaseMessage = msg
    if used:
        final = HumanMessage(
            content=[
                {"type": "text", "text": msg.content},
                *[
                    image_path_block(
                        u["path"],
                        n=u["n"],
                        chunk_id=u["chunk_id"],
                        paper_id=u["paper_id"],
                    )
                    for u in used
                ],
            ]
        )

    return [
        SystemMessage(content=SYSTEM_PROMPT),
        *_history_messages(history or []),
        final,
    ]


def _format_history_for_rewrite(history: list[dict]) -> str:
    recent = history[-2 * REWRITE_MAX_TURNS :]
    lines = []
    for m in recent:
        role = "User" if m.get("role") == "user" else "Assistant"
        content = (m.get("content") or "").strip()
        if len(content) > REWRITE_MAX_CHARS_PER_MSG:
            content = content[:REWRITE_MAX_CHARS_PER_MSG].rstrip() + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _clean_rewrite(raw: str, original: str) -> str:
    """Normalize model output; fall back to the original query if it looks off."""
    text = raw.strip()
    if not text:
        return original
    text = text.splitlines()[0].strip()  # models sometimes add explanations below
    text = re.sub(r"^(standalone query|query)\s*:\s*", "", text, flags=re.I)
    text = text.strip(" \"'`\u201c\u201d")
    if not text or len(text) > REWRITE_MAX_QUERY_CHARS:
        return original
    return text


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------
def build_rag_graph(
    embedder: QueryEmbedder,
    client: QdrantClient,
    llm: ChatOllama,
    reranker: CohereReranker,
    use_images: bool = True,
    rewrite_llm: Optional[BaseChatModel] = None,
) -> CompiledStateGraph:
    if use_images and not isinstance(llm, PathImageChatOllama):
        raise TypeError(
            "use_images=True needs a PathImageChatOllama (app.services.vision_llm); "
            "pass use_images=False to keep the text-only behavior."
        )
    images_root = llm.images_root if isinstance(llm, PathImageChatOllama) else None
    chain = llm | StrOutputParser()
    chain = llm | StrOutputParser()
    rewrite_chain = (
        REWRITE_PROMPT | (rewrite_llm or llm) | StrOutputParser()
    ).with_config(run_name="rewrite_query")

    async def rewrite(state: RAGState, config: RunnableConfig) -> dict:
        query = state["query"]
        history = state.get("history") or []
        if not history:
            return {"rewritten_query": query}
        try:
            raw = await rewrite_chain.ainvoke(
                {
                    "history": _format_history_for_rewrite(history),
                    "question": query,
                },
                config=config,  # propagates the Langfuse callback
            )
        except Exception:
            logger.exception("Query rewrite failed; using original query")
            return {"rewritten_query": query}

        rewritten = _clean_rewrite(raw, query)
        logger.info("query rewrite: %r -> %r", query, rewritten)
        return {"rewritten_query": rewritten}

    async def retrieve(state: RAGState) -> dict:
        top_k = state["top_k"]
        search_query = state.get("rewritten_query") or state["query"]
        candidate_k = max(RERANK_CANDIDATES, top_k)
        try:
            candidates = await asyncio.to_thread(
                retrieve_top_k,
                query=search_query,
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
                query=search_query, chunks=candidates, top_n=top_k
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
            return {
                "answer": NO_DOCS_ANSWER,
                "images_used": [],
                "images_skipped": [],
                "image_fallback": False,
            }

        history = state.get("history") or []
        used: list[dict] = []
        skipped: list[dict] = []

        if use_images and images_root is not None:
            used, skipped = _collect_images(chunks, images_root)

        for s in skipped:
            logger.warning("image skipped: %s", s)

        image_fallback = False

        try:
            with propagate_attributes(
                tags=["with_images" if used else "text_only"],
                metadata={
                    "images_used": str(bool(used)).lower(),
                    "num_images": str(len(used)),
                    "image_status": "used" if used else "text_only",
                    "image_paths": ", ".join(u["path"] for u in used),
                    "image_chunk_ids": ", ".join(str(u["chunk_id"]) for u in used),
                    "image_paper_ids": ", ".join(
                        sorted({str(u["paper_id"]) for u in used})
                    ),
                    "images_skipped": str(len(skipped)),
                },
            ):
                answer = await chain.ainvoke(
                    _build_messages(chunks, state["query"], used, history),
                    config=_trace_config(config, used, skipped),
                )

        except Exception as e:
            if not used:
                raise GenerationError(str(e)) from e

            logger.exception("LLM call with images failed; retrying text-only")
            image_fallback = True
            used = []

            try:
                with propagate_attributes(
                    tags=["text_only", "image_fallback"],
                    metadata={
                        "images_used": "false",
                        "num_images": "0",
                        "image_status": "fallback_after_image_error",
                        "image_fallback": "true",
                        "images_skipped": str(len(skipped)),
                    },
                ):
                    answer = await chain.ainvoke(
                        _build_messages(chunks, state["query"], used, history),
                        config=_trace_config(config, used, skipped, fallback=True),
                    )
            except Exception as e2:
                raise GenerationError(str(e2)) from e2

        return {
            "answer": answer,
            "images_used": used,
            "images_skipped": skipped,
            "image_fallback": image_fallback,
        }

    builder = StateGraph(RAGState)
    builder.add_node("rewrite", rewrite)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_conditional_edges(
        START,
        lambda s: "rewrite" if s.get("history") else "retrieve",
        {"rewrite": "rewrite", "retrieve": "retrieve"},
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()
