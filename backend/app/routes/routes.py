from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.controllers.controller import (
    handle_rag_answer,
)
from app.db.db import get_session
from app.db.models import User


def get_rag_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.rag_graph


def get_title_llm(request: Request) -> BaseChatModel:
    return request.app.state.title_llm


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class SourceItem(BaseModel):
    score: float
    chunk_id: Optional[str] = None
    paper_id: Optional[str] = None
    text: Optional[str] = None
    payload: dict


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's question")
    conversation_id: Optional[UUID] = Field(
        None,
        description=(
            "Existing conversation to continue (not supported yet); "
            "omit to start a new one"
        ),
    )


class QueryResponse(BaseModel):
    conversation_id: UUID
    query: str
    answer: str
    sources: List[SourceItem]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
router = APIRouter()


@router.post("/answer", response_model=QueryResponse)
async def answer(
    request: QueryRequest,
    graph: CompiledStateGraph = Depends(get_rag_graph),
    title_llm: BaseChatModel = Depends(get_title_llm),
    user: User = Depends(get_current_user),
    # Same dependency as get_current_user uses, so FastAPI hands both the
    # same session instance within a request.
    db: AsyncSession = Depends(get_session),
):
    return await handle_rag_answer(
        query=request.query,
        conversation_id=request.conversation_id,
        graph=graph,
        title_llm=title_llm,
        user_id=user.id,
        db=db,
    )
