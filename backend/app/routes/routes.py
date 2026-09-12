from typing import List, Optional

from fastapi import APIRouter, Depends
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from app.controllers.controller import (
    get_embedder,
    get_llm,
    get_qdrant_client,
    handle_rag_answer,
)
from app.services.retrieve_from_qdrant import QueryEmbedder


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
    top_k: int = Field(
        5, ge=1, le=20, description="Number of chunks to retrieve for context"
    )
    paper_id: Optional[str] = Field(None, description="Filter to a specific paper ID")
    has_picture: Optional[bool] = Field(
        None, description="Filter to chunks containing pictures"
    )


class QueryResponse(BaseModel):
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
    embedder: QueryEmbedder = Depends(get_embedder),
    client: QdrantClient = Depends(get_qdrant_client),
    llm: ChatGoogleGenerativeAI = Depends(get_llm),
):
    """
    Retrieves top-k chunks and uses Gemini to generate an answer.
    """
    return await handle_rag_answer(
        query=request.query,
        top_k=request.top_k,
        paper_id=request.paper_id,
        has_picture=request.has_picture,
        embedder=embedder,
        client=client,
        llm=llm,
    )
