"""Controller for the streaming RAG endpoint (Server-Sent Events).

Flow of one request
-------------------
1. `prepare_turn` (runs in the route, BEFORE the response starts): validates the
   query and the conversation, loads history, ends the read transaction. Errors
   here are normal HTTP errors (400 / 404).
2. `stream_rag_answer` returns the SSE body. A producer task does the real work
   (compaction -> graph -> persistence) and pushes events into a queue; the
   generator only reads the queue. The producer is a separate task so the
   Langfuse/OpenTelemetry `with propagate_attributes(...)` blocks always enter
   and exit in the same task, even if the client disconnects mid-stream.

Events (all `data:` payloads are JSON)
--------------------------------------
title    {"title": str}                       new conversations only
sources  [{score, chunk_id, paper_id, text, payload}]   right after rerank
token    {"text": str}                        generate node only
reset    {}                                   image-fallback retry: drop partial text
done     {"conversation_id", "title", "answer"}  sent AFTER the DB commit; `answer`
                                              is authoritative (replace streamed text)
error    {"detail": str}                      terminal; nothing was saved
": keepalive" comment every KEEPALIVE_SECONDS while waiting.

On client disconnect the producer is cancelled and nothing is persisted.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Optional, cast
from uuid import UUID, uuid4

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from langchain_core.language_models import BaseChatModel
from langfuse import propagate_attributes
from langfuse.langchain import CallbackHandler
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.db import AsyncSessionLocal
from app.db.models import MessageRole
from app.graph.rag_graph import GenerationError, RetrievalError
from app.services.conversation_service import (
    add_message,
    build_title,
    create_conversation,
    get_history,
    get_owned_conversation,
    slim_sources,
    touch_conversation,
)
from app.services.summary_service import compact_history
from app.services.title_service import generate_title

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
KEEPALIVE_SECONDS = 15

_END = object()  # queue sentinel
Emit = Callable[[str, Any], None]


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------
def sse(event: str, data: Any) -> str:
    """One SSE frame. JSON on a single line, so newlines in tokens can't break
    the framing."""
    payload = json.dumps(jsonable_encoder(data), ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _public_source(c: dict) -> dict:
    """Same shape the old QueryResponse.sources (SourceItem) exposed."""
    return {
        "score": c.get("score"),
        "chunk_id": c.get("chunk_id"),
        "paper_id": c.get("paper_id"),
        "text": c.get("text"),
        "payload": c.get("payload") or {},
    }


# ---------------------------------------------------------------------------
# Step 1: everything that must fail as a plain HTTP error
# ---------------------------------------------------------------------------
@dataclass
class TurnContext:
    query: str
    user_id: int
    conversation_id: UUID
    is_new: bool
    history: list[dict]
    existing_title: Optional[str]
    summary: Optional[str]
    summary_upto_id: Optional[int]


async def prepare_turn(
    *,
    query: str,
    conversation_id: Optional[UUID],
    user_id: int,
    db: AsyncSession,
) -> TurnContext:
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    is_new = conversation_id is None
    history: list[dict] = []
    existing_title: Optional[str] = None
    summary: Optional[str] = None
    summary_upto_id: Optional[int] = None

    if conversation_id is None:
        # Generated up front so Langfuse's session_id and the DB primary key
        # are the same value.
        conversation_id = uuid4()
    else:
        conversation = await get_owned_conversation(
            db, conversation_id=conversation_id, user_id=user_id
        )
        if conversation is None:
            # Same response for "doesn't exist" and "not yours".
            raise HTTPException(status_code=404, detail="Conversation not found")
        existing_title = conversation.title
        summary = conversation.summary
        summary_upto_id = conversation.summary_upto_message_id
        # Loaded before the new user message is persisted, so the history
        # never contains the current query.
        history = await get_history(db, conversation_id, after_id=summary_upto_id)

    # End the read transaction opened by get_current_user so no pool connection
    # is held while the stream runs. Everything read above is plain Python.
    await db.commit()

    return TurnContext(
        query=query,
        user_id=user_id,
        conversation_id=conversation_id,
        is_new=is_new,
        history=history,
        existing_title=existing_title,
        summary=summary,
        summary_upto_id=summary_upto_id,
    )


# ---------------------------------------------------------------------------
# Step 2: producer (runs in its own task)
# ---------------------------------------------------------------------------
async def _make_title(llm: BaseChatModel, query: str, emit: Emit) -> str:
    try:
        title = await generate_title(llm, query)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Title generation failed; using truncated query")
        title = None
    title = title or build_title(query)
    emit("title", {"title": title})
    return title


async def _stream_graph(
    ctx: TurnContext,
    graph: CompiledStateGraph,
    handler: CallbackHandler,
    summary: Optional[str],
    history: list[dict],
    emit: Emit,
) -> tuple[list[dict], str]:
    chunks: list[dict] = []
    answer: Optional[str] = None
    current_run: Optional[str] = None  # message id == one LLM run
    streamed = False

    with propagate_attributes(
        trace_name="rag-answer",
        user_id=str(ctx.user_id),
        session_id=str(ctx.conversation_id),
        tags=["rag"],
        metadata={
            "top_k": str(DEFAULT_TOP_K),
            "history_len": str(len(history)),
            "has_summary": str(bool(summary)).lower(),
            "streaming": "true",
        },
    ):
        async for part in graph.astream(
            {
                "query": ctx.query,
                "top_k": DEFAULT_TOP_K,
                "paper_id": None,
                "has_picture": None,
                "history": history,
                "history_summary": summary,
            },
            config={"callbacks": [handler]},
            stream_mode=["messages", "updates"],
        ):
            # With a list of stream modes each item is a (mode, data) tuple, but
            # astream is typed as `dict | Any`, so the checker would otherwise
            # infer `data` as str.
            mode, data = cast(tuple[str, Any], part)
            if mode == "messages":
                msg, meta = data
                # Skip the rewrite node's tokens.
                if meta.get("langgraph_node") != "generate":
                    continue
                text = msg.content
                if not isinstance(text, str) or not text:
                    continue
                # A new message id after tokens were already sent means the
                # image attempt failed and `generate` retried text-only.
                if streamed and msg.id != current_run:
                    emit("reset", {})
                current_run = msg.id
                streamed = True
                emit("token", {"text": text})

            elif mode == "updates":
                for node, update in data.items():
                    if not isinstance(update, dict):
                        continue
                    if node == "retrieve":
                        chunks = update.get("chunks") or []
                        emit("sources", [_public_source(c) for c in chunks])
                    elif node == "generate":
                        answer = update.get("answer")

    if answer is None:
        raise GenerationError("The graph finished without producing an answer")
    return chunks, answer


async def _persist_turn(
    ctx: TurnContext,
    *,
    title: str,
    answer: str,
    chunks: list[dict],
    trace_id: Optional[str],
) -> None:
    """One transaction per turn: [conversation +] user message + assistant message.
    Fresh session on purpose: the request-scoped one belongs to FastAPI's
    dependency lifecycle, which we don't want to depend on here."""
    async with AsyncSessionLocal() as db:
        if ctx.is_new:
            await create_conversation(
                db,
                conversation_id=ctx.conversation_id,
                user_id=ctx.user_id,
                title=title,
            )
        await add_message(
            db,
            conversation_id=ctx.conversation_id,
            role=MessageRole.user,
            content=ctx.query,
        )
        await add_message(
            db,
            conversation_id=ctx.conversation_id,
            role=MessageRole.assistant,
            content=answer,
            sources=slim_sources(chunks),
            trace_id=trace_id,
        )
        if not ctx.is_new:
            await touch_conversation(db, ctx.conversation_id)
        await db.commit()
        # On any exception the session closes without commit -> rollback.


async def _produce(
    ctx: TurnContext,
    graph: CompiledStateGraph,
    title_llm: BaseChatModel,
    summary_llm: BaseChatModel,
    queue: "asyncio.Queue[Any]",
) -> None:
    def emit(event: str, data: Any = None) -> None:
        queue.put_nowait((event, {} if data is None else data))

    title_task: Optional[asyncio.Task] = None
    try:
        summary, history = ctx.summary, ctx.history

        if not ctx.is_new:
            # Cap the context: fold the oldest turns into the summary (persisted)
            # and keep the recent tail verbatim. Never raises. Own trace, same
            # session_id.
            async with AsyncSessionLocal() as db:
                with propagate_attributes(
                    trace_name="summarize-history",
                    user_id=str(ctx.user_id),
                    session_id=str(ctx.conversation_id),
                    tags=["summary"],
                ):
                    summary, history = await compact_history(
                        summary_llm,
                        db,
                        conversation_id=ctx.conversation_id,
                        summary=summary,
                        summary_upto_id=ctx.summary_upto_id,
                        history=history,
                        config={"callbacks": [CallbackHandler()]},
                    )

        handler = CallbackHandler()
        if ctx.is_new:
            title_task = asyncio.create_task(_make_title(title_llm, ctx.query, emit))

        chunks, answer = await _stream_graph(
            ctx, graph, handler, summary, history, emit
        )

        title = (
            await title_task if title_task is not None else (ctx.existing_title or "")
        )

        # Verify this attribute exists on your Langfuse v4 CallbackHandler;
        # if not, trace_id stays None and we log it.
        trace_id = getattr(handler, "last_trace_id", None)
        if trace_id is None:
            logger.warning(
                "No Langfuse trace id available for conversation %s",
                ctx.conversation_id,
            )

        try:
            await _persist_turn(
                ctx, title=title, answer=answer, chunks=chunks, trace_id=trace_id
            )
        except SQLAlchemyError:
            logger.exception("Failed to persist conversation %s", ctx.conversation_id)
            emit("error", {"detail": "Failed to save conversation"})
            return

        emit(
            "done",
            {
                "conversation_id": ctx.conversation_id,
                "title": title,
                "answer": answer,
            },
        )

    except asyncio.CancelledError:
        logger.info("Stream cancelled (client disconnected?): %s", ctx.conversation_id)
        raise
    except RetrievalError as e:
        emit("error", {"detail": f"Retrieval failed: {e}"})
    except GenerationError as e:
        emit("error", {"detail": f"LLM generation failed: {e}"})
    except Exception:
        logger.exception("Unexpected error in stream for %s", ctx.conversation_id)
        emit("error", {"detail": "Internal error"})
    finally:
        if title_task is not None and not title_task.done():
            title_task.cancel()
        queue.put_nowait(_END)


# ---------------------------------------------------------------------------
# Step 3: the SSE body
# ---------------------------------------------------------------------------
async def stream_rag_answer(
    ctx: TurnContext,
    graph: CompiledStateGraph,
    title_llm: BaseChatModel,
    summary_llm: BaseChatModel,
) -> AsyncIterator[str]:
    queue: asyncio.Queue[Any] = asyncio.Queue()
    producer = asyncio.create_task(_produce(ctx, graph, title_llm, summary_llm, queue))
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if item is _END:
                break
            event, data = item
            yield sse(event, data)
    finally:
        # Client gone (or we're done): make sure the producer stops. Its own
        # `finally` cancels the title task.
        if not producer.done():
            producer.cancel()
