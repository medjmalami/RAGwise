from __future__ import annotations

import asyncio
import logging
import re
import time
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
from langfuse import get_client, propagate_attributes
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import QdrantClient, models

from app.services.rerank import CohereReranker
from app.services.retrieve_from_qdrant import (
    DEFAULT_COLLECTION,
    QueryEmbedder,
    embed_query,
    retrieve_top_k,
)
from app.services.semantic_cache import (
    CacheOptions,
    SemanticCache,
    fingerprint,
    validate_payload,
    write_skip_reason,
)
from app.services.vision_llm import PathImageChatOpenAI, image_path_block

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
context, just say you don't know. Don't try to make up an answer.

Start directly with the answer; do not say "Based on the provided text" or \
describe the prompt or context. Format lists as Markdown: leave a blank line \
before each list, put one flat "- " item on each line, and do not combine or \
nest list items on one line. Use plain arrows (-> or →), not LaTeX, for simple \
flows. Cite supporting retrieved passages with [1], [2], etc. in their order in \
the context. When an attached image is relevant, describe its visible content \
and cite that passage; do not refer to it only as "Image N"."""

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

REWRITE_MAX_SUMMARY_CHARS = 1200
SUMMARY_NOTE = (
    "\n\nSummary of the earlier part of this conversation (older turns, condensed). "
    "Use it to resolve references to earlier turns; factual claims about papers "
    "must still come from the retrieved context.\n{summary}"
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
    cache_status: str  # hit / miss / error / skipped (lookup outcome)
    cache_score: Optional[float]
    cache_embedding: Optional[dict]  # dense + sparse, reused by retrieval and write
    cache_write_status: str
    cache_skip_reason: Optional[str]
    cache_created_at: Optional[int]
    rewritten_query: str  # standalone query used for retrieval + rerank
    candidates: list[dict]  # slim metadata of the 50 hybrid candidates (pre-rerank)
    chunks: list[dict]  # Cohere top_k, full chunks (text untouched by the image step)
    answer: str
    rerank_fallback: bool
    images_used: list[dict]  # [{n, chunk_id, paper_id, path}] -- paths only, no bytes
    images_skipped: list[dict]  # [{chunk_id, path, reason}]
    image_fallback: bool  # True if the LLM call with images failed -> text-only retry
    history_summary: Optional[str]


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
            # images are sent as content blocks after the text; this label ties each image to its chunk.
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
    summary: Optional[str] = None,
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
        SystemMessage(
            content=SYSTEM_PROMPT
            + (SUMMARY_NOTE.format(summary=summary.strip()) if summary else "")
        ),
        *_history_messages(history or []),
        final,
    ]


def _format_history_for_rewrite(
    history: list[dict], summary: Optional[str] = None
) -> str:
    recent = history[-2 * REWRITE_MAX_TURNS :]
    lines = []
    if summary:
        s = summary.strip()
        if len(s) > REWRITE_MAX_SUMMARY_CHARS:
            s = s[:REWRITE_MAX_SUMMARY_CHARS].rstrip() + "..."
        lines.append(f"[Summary of earlier messages]\n{s}\n[Most recent messages]")
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
    llm: BaseChatModel,
    reranker: CohereReranker,
    use_images: bool = True,
    rewrite_llm: Optional[BaseChatModel] = None,
    use_cache: bool = False,
    cache: Optional[SemanticCache] = None,
    cache_options: Optional[CacheOptions] = None,
) -> CompiledStateGraph:
    if use_images and not isinstance(llm, PathImageChatOpenAI):
        raise TypeError(
            "use_images=True needs a PathImageChatOpenAI (app.services.vision_llm); "
            "pass use_images=False to keep the text-only behavior."
        )
    images_root = llm.images_root if isinstance(llm, PathImageChatOpenAI) else None
    chain = llm | StrOutputParser()
    rewrite_chain = (
        REWRITE_PROMPT | (rewrite_llm or llm) | StrOutputParser()
    ).with_config(run_name="rewrite_query")

    options = cache.options if cache is not None else (cache_options or CacheOptions())
    cache_store = cache or (SemanticCache(client, options) if use_cache else None)
    # Explicit allowlist: never fingerprint credentials or serialize the entire client.
    cache_config = {
        "schema": 1,
        "corpus": options.corpus_version,
        "collection": DEFAULT_COLLECTION,
        "embedder": [embedder.model_name, embedder.max_length],
        "llm_class": type(llm).__name__,
        "llm": {
            key: getattr(llm, key, None)
            for key in (
                "model_name",
                "openai_api_base",
                "temperature",
                "max_tokens",
                "model_kwargs",
                "extra_body",
                "reasoning_effort",
                "seed",
                "top_p",
            )
        },
        "reranker": reranker.model,
        "candidates": RERANK_CANDIDATES,
        "prompts": [SYSTEM_PROMPT, HUMAN_PROMPT.model_dump(), IMAGE_NOTE],
        "images": [use_images, MAX_IMAGES_PER_CHUNK, MAX_IMAGES_TOTAL],
    }

    def version(state: RAGState) -> str:
        return fingerprint(
            {
                **cache_config,
                "top_k": state["top_k"],
                "paper_id": state.get("paper_id"),
                "has_picture": state.get("has_picture"),
            }
        )

    def trace_cache(update: dict) -> dict:
        metadata = {
            key: str(value)
            for key, value in update.items()
            if key.startswith("cache_")
            and key != "cache_embedding"
            and value is not None
        }
        metadata["use_cache"] = str(use_cache).lower()
        if update.get("cache_created_at") is not None:
            metadata["cache_age_seconds"] = str(
                max(0, int(time.time()) - update["cache_created_at"])
            )
        tags = ["cache:on" if use_cache else "cache:off"]
        if update.get("cache_status"):
            tags.append("cache:" + update["cache_status"])
        if update.get("cache_status") == "hit":
            used = update.get("images_used", [])
            tags.append("with_images" if used else "text_only")
            metadata.update(
                images_used=str(bool(used)).lower(),
                num_images=str(len(used)),
                image_paths=", ".join(i["path"] for i in used),
            )
        # A child observation makes propagation visible even on hits (no LLM call).
        try:
            with propagate_attributes(tags=tags, metadata=metadata):
                with get_client().start_as_current_observation(
                    name="semantic-cache", as_type="span", metadata=metadata
                ):
                    pass
        except Exception:
            logger.exception("Cache telemetry failed")
        return update

    async def read_cache(state: RAGState) -> dict:
        update: dict = {
            "cache_status": "miss",
            "cache_score": None,
            "cache_embedding": None,
            "cache_write_status": "pending",
            "cache_skip_reason": None,
            "cache_created_at": None,
            "rewritten_query": state["query"],
        }
        try:
            # Compute sparse weights too: a miss can reuse the complete hybrid embedding.
            embedded = await asyncio.to_thread(embed_query, state["query"], embedder)
            update["cache_embedding"] = embedded
            assert cache_store is not None
            hit = await cache_store.read(
                embedded["dense"], version(state), state["top_k"]
            )
            if hit is not None:
                update["cache_score"] = float(hit.score)
                try:
                    restored = validate_payload(hit.payload or {}, images_root)
                    created_at = int((hit.payload or {})["created_at"])
                except (KeyError, TypeError, ValueError, OSError):
                    update["cache_skip_reason"] = "invalid_payload"
                else:
                    update.update(restored)
                    update.update(
                        cache_status="hit",
                        cache_write_status="skipped",
                        cache_created_at=created_at,
                    )
        except Exception:
            logger.exception("Cache read failed; continuing with retrieval")
            update["cache_status"] = "error"
        return trace_cache(update)

    async def write_cache(state: RAGState) -> dict:
        reason = write_skip_reason(state)
        embedded = state.get("cache_embedding")
        if reason or not embedded:
            return trace_cache(
                {
                    "cache_write_status": "skipped",
                    "cache_skip_reason": reason or "no_embedding",
                }
            )
        try:
            assert cache_store is not None
            await cache_store.write(embedded["dense"], version(state), dict(state))
            return trace_cache({"cache_write_status": "written"})
        except Exception:
            logger.exception("Cache write failed; keeping generated answer")
            return trace_cache({"cache_write_status": "error"})

    async def rewrite(state: RAGState, config: RunnableConfig) -> dict:
        query = state["query"]
        history = state.get("history") or []
        summary = state.get("history_summary")
        if not history and not summary:
            return {"rewritten_query": query}
        try:
            raw = await rewrite_chain.ainvoke(
                {
                    "history": _format_history_for_rewrite(history, summary),
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
        bypass = not use_cache or bool(
            state.get("history") or state.get("history_summary")
        )
        cache_update = (
            trace_cache(
                {
                    "cache_status": "skipped",
                    "cache_score": None,
                    "cache_embedding": None,
                    "cache_write_status": "skipped",
                    "cache_skip_reason": "follow_up" if use_cache else "disabled",
                }
            )
            if bypass
            else {}
        )
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
                embedded=state.get("cache_embedding")
                if use_cache
                and not (state.get("history") or state.get("history_summary"))
                else None,
                query_filter=_build_filter(
                    state.get("paper_id"), state.get("has_picture")
                ),
            )
        except Exception as e:
            raise RetrievalError(str(e)) from e
        if not candidates:
            return {
                **cache_update,
                "candidates": [],
                "chunks": [],
                "rerank_fallback": False,
            }
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
            **cache_update,
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
        summary = state.get("history_summary")
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
                    _build_messages(chunks, state["query"], used, history, summary),
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
                        _build_messages(chunks, state["query"], used, history, summary),
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
    if use_cache:
        builder.add_node("read_cache", read_cache)
        builder.add_node("write_cache", write_cache)
        builder.add_conditional_edges(
            "read_cache",
            lambda s: END if s.get("cache_status") == "hit" else "retrieve",
            {END: END, "retrieve": "retrieve"},
        )
        builder.add_edge("write_cache", END)
    builder.add_conditional_edges(
        START,
        lambda s: (
            "rewrite"
            if (s.get("history") or s.get("history_summary"))
            else ("read_cache" if use_cache else "retrieve")
        ),
        {
            "rewrite": "rewrite",
            "retrieve": "retrieve",
            **({"read_cache": "read_cache"} if use_cache else {}),
        },
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "generate")
    if use_cache:
        builder.add_conditional_edges(
            "generate",
            lambda s: (
                END if (s.get("history") or s.get("history_summary")) else "write_cache"
            ),
            {END: END, "write_cache": "write_cache"},
        )
    else:
        builder.add_edge("generate", END)
    return builder.compile()
