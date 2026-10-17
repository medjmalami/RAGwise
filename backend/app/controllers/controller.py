import logging
from typing import Optional
from uuid import UUID, uuid4

from fastapi import HTTPException, Request
from langfuse import propagate_attributes
from langfuse.langchain import CallbackHandler
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MessageRole
from app.graph.rag_graph import GenerationError, RetrievalError
from app.services.conversation_service import (
    add_message,
    build_title,
    create_conversation,
    slim_sources,
)

logger = logging.getLogger(__name__)


def get_rag_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.rag_graph


DEFAULT_TOP_K = 5


async def handle_rag_answer(
    query: str,
    conversation_id: Optional[UUID],
    graph: CompiledStateGraph,
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
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")
    except GenerationError as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

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
            title=build_title(query),
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
        "query": query,
        "answer": answer,
        "sources": chunks,
    }
