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
    slim_sources,
)
from app.services.title_service import generate_title

logger = logging.getLogger(__name__)


DEFAULT_TOP_K = 5


async def handle_rag_answer(
    query: str,
    conversation_id: Optional[UUID],
    graph: CompiledStateGraph,
    title_llm: BaseChatModel,
    user_id: int,
    db: AsyncSession,
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Sprint 1 covers new conversations only. Sprint 2 replaces this with
    # ownership check + history loading.
    if conversation_id is not None:
        raise HTTPException(
            status_code=501,
            detail="Continuing an existing conversation is not supported yet",
        )

    # Generated up front so Langfuse's session_id and the DB primary key are
    # the same value.
    conversation_id = uuid4()

    # get_current_user already opened a (read-only) transaction on this
    # session, which holds a pool connection. End it before the slow graph
    # call. expire_on_commit=False keeps loaded objects usable.
    await db.commit()

    handler = CallbackHandler()
    title_task = asyncio.create_task(generate_title(title_llm, query))
    try:
        with propagate_attributes(
            trace_name="rag-answer",
            user_id=str(user_id),
            session_id=str(conversation_id),
            tags=["rag"],
            metadata={"top_k": str(DEFAULT_TOP_K)},
        ):
            result = await graph.ainvoke(
                {
                    "query": query,
                    "top_k": DEFAULT_TOP_K,
                    "paper_id": None,
                    "has_picture": None,
                },
                config={"callbacks": [handler]},
            )
    except RetrievalError as e:
        title_task.cancel()
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")
    except GenerationError as e:
        title_task.cancel()
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")
    except BaseException:
        title_task.cancel()
        raise

    title = await title_task
    answer = result["answer"]
    chunks = result.get("chunks", [])

    # Verify this attribute exists on your Langfuse v4 CallbackHandler;
    # if not, trace_id stays None and we log it.
    trace_id = getattr(handler, "last_trace_id", None)
    if trace_id is None:
        logger.warning(
            "No Langfuse trace id available for conversation %s", conversation_id
        )

    # One transaction per turn: conversation + user message + assistant message.
    try:
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
