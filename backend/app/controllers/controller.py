import asyncio
import logging
from typing import Optional
from uuid import UUID, uuid4

from fastapi import HTTPException
from langchain_core.language_models import BaseChatModel
from langfuse import propagate_attributes
from langfuse.langchain import CallbackHandler
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MessageRole
from app.graph.rag_graph import GenerationError, RetrievalError
from app.services.conversation_service import (
    add_message,
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


async def handle_rag_answer(
    query: str,
    conversation_id: Optional[UUID],
    graph: CompiledStateGraph,
    title_llm: BaseChatModel,
    summary_llm: BaseChatModel,
    user_id: int,
    db: AsyncSession,
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    is_new = conversation_id is None
    history: list[dict] = []
    existing_title: Optional[str] = None
    summary: Optional[str] = None
    summary_upto_id: Optional[int] = None

    if is_new:
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
        # never contains the current query. Messages up to summary_upto_id are
        # already folded into `summary`, so only the rest is loaded.
        history = await get_history(db, conversation_id, after_id=summary_upto_id)

    # get_current_user already opened a (read-only) transaction on this
    # session, which holds a pool connection. End it before the slow LLM calls
    # below. Everything read above is already plain Python (str / int / list[dict]).
    await db.commit()

    if not is_new:
        # Cap the context: if summary + history is over the budget, fold the
        # oldest turns into the summary (persisted) and keep the recent tail
        # verbatim. Never raises; on failure it returns the inputs unchanged.
        # Own trace, same session_id, so it shows up in the same Langfuse session.
        with propagate_attributes(
            trace_name="summarize-history",
            user_id=str(user_id),
            session_id=str(conversation_id),
            tags=["summary"],
        ):
            summary, history = await compact_history(
                summary_llm,
                db,
                conversation_id=conversation_id,
                summary=summary,
                summary_upto_id=summary_upto_id,
                history=history,
                config={"callbacks": [CallbackHandler()]},
            )

    handler = CallbackHandler()
    # Only new conversations need a title; continuations keep theirs.
    title_task = (
        asyncio.create_task(generate_title(title_llm, query)) if is_new else None
    )

    def _cancel_title() -> None:
        if title_task is not None:
            title_task.cancel()

    try:
        with propagate_attributes(
            trace_name="rag-answer",
            user_id=str(user_id),
            session_id=str(conversation_id),
            tags=["rag"],
            metadata={
                "top_k": str(DEFAULT_TOP_K),
                "history_len": str(len(history)),
                "has_summary": str(bool(summary)).lower(),
            },
        ):
            result = await graph.ainvoke(
                {
                    "query": query,
                    "top_k": DEFAULT_TOP_K,
                    "paper_id": None,
                    "has_picture": None,
                    "history": history,
                    "history_summary": summary,
                },
                config={"callbacks": [handler]},
            )
    except RetrievalError as e:
        _cancel_title()
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")
    except GenerationError as e:
        _cancel_title()
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")
    except BaseException:
        _cancel_title()
        raise

    if title_task is not None:
        title = await title_task
    else:
        # Column is nullable; every conversation created through this endpoint
        # has a title, so this is only a guard for QueryResponse.title: str.
        title = existing_title or ""

    answer = result["answer"]
    chunks = result.get("chunks", [])

    # Verify this attribute exists on your Langfuse v4 CallbackHandler;
    # if not, trace_id stays None and we log it.
    trace_id = getattr(handler, "last_trace_id", None)
    if trace_id is None:
        logger.warning(
            "No Langfuse trace id available for conversation %s", conversation_id
        )

    # One transaction per turn: [conversation +] user message + assistant message.
    try:
        if is_new:
            await create_conversation(
                db,
                conversation_id=conversation_id,
                user_id=user_id,
                title=title,
            )
        await add_message(
            db,
            conversation_id=conversation_id,
            role=MessageRole.user,
            content=query,
        )
        await add_message(
            db,
            conversation_id=conversation_id,
            role=MessageRole.assistant,
            content=answer,
            sources=slim_sources(chunks),
            trace_id=trace_id,
        )
        if not is_new:
            await touch_conversation(db, conversation_id)
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        logger.exception("Failed to persist conversation %s", conversation_id)
        raise HTTPException(status_code=500, detail="Failed to save conversation")

    return {
        "conversation_id": conversation_id,
        "title": title,
        "query": query,
        "answer": answer,
        "sources": chunks,
    }
