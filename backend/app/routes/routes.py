from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.controllers.controller import (
    get_rag_graph,
    handle_rag_answer,
)
from app.db.db import get_session
from app.db.models import User


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
    user: User = Depends(get_current_user),
    # Same dependency as get_current_user uses, so FastAPI hands both the
    # same session instance within a request.
    db: AsyncSession = Depends(get_session),
):
    return await handle_rag_answer(
        query=request.query,
        conversation_id=request.conversation_id,
        graph=graph,
        user_id=user.id,
        db=db,
    )
