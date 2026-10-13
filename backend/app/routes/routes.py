from typing import List, Optional
from uuid import UUID  # new import

from fastapi import APIRouter, Depends
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from app.controllers.controller import (
    get_rag_graph,
    handle_rag_answer,
)


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
        None, description="Existing conversation to continue; omit to start a new one"
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
):
    return await handle_rag_answer(
        query=request.query,
        conversation_id=request.conversation_id,
        graph=graph,
    )
