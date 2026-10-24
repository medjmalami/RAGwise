from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.controllers.controller import prepare_turn, stream_rag_answer
from app.db.db import get_session
from app.db.models import User


def get_rag_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.rag_graph


def get_title_llm(request: Request) -> BaseChatModel:
    return request.app.state.title_llm


def get_summary_llm(request: Request) -> BaseChatModel:
    return request.app.state.summary_llm


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's question")
    conversation_id: Optional[UUID] = Field(
        None,
        description=(
            "Existing conversation to continue; omit to start a new one. "
            "Returns 404 if it doesn't exist or isn't yours."
        ),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
router = APIRouter()

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    # Tell nginx-style proxies not to buffer the stream.
    "X-Accel-Buffering": "no",
}


@router.post(
    "/answer",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": (
                "Server-Sent Events stream. Events (JSON `data`): "
                "`title` {title} (new conversations only); "
                "`sources` [{score, chunk_id, paper_id, text, payload}]; "
                "`token` {text}; "
                "`reset` {} (discard the text streamed so far); "
                "`done` {conversation_id, title, answer} (sent after the turn is "
                "saved; `answer` is authoritative); "
                "`error` {detail} (terminal, nothing saved). "
                "Lines starting with ':' are keepalives."
            ),
            "content": {"text/event-stream": {}},
        },
        400: {"description": "Empty query"},
        404: {"description": "Conversation not found"},
    },
)
async def answer(
    request: QueryRequest,
    graph: CompiledStateGraph = Depends(get_rag_graph),
    title_llm: BaseChatModel = Depends(get_title_llm),
    user: User = Depends(get_current_user),
    # Same dependency as get_current_user uses, so FastAPI hands both the
    # same session instance within a request.
    db: AsyncSession = Depends(get_session),
    summary_llm: BaseChatModel = Depends(get_summary_llm),
):
    # 400 / 404 are raised here, before the stream starts, so they stay real
    # HTTP status codes.
    ctx = await prepare_turn(
        query=request.query,
        conversation_id=request.conversation_id,
        user_id=user.id,
        db=db,
    )
    return StreamingResponse(
        stream_rag_answer(ctx, graph, title_llm, summary_llm),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
