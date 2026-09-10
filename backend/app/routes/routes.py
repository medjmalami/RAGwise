from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from app.controllers.controller import get_embedder, get_qdrant_client, handle_retrieve
from app.services.retrieve_from_qdrant import QueryEmbedder


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class SearchResultItem(BaseModel):
    score: float
    chunk_id: Optional[str] = None
    paper_id: Optional[str] = None
    text: Optional[str] = None
    payload: dict


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]


class SearchRequest(BaseModel):
    query: str = Field(..., description="The user's search query")
    top_k: int = Field(5, ge=1, le=50, description="Number of chunks to return")
    paper_id: Optional[str] = Field(None, description="Filter to a specific paper ID")
    has_picture: Optional[bool] = Field(
        None, description="Filter to chunks containing pictures"
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
router = APIRouter()


@router.post("/retrieve", response_model=SearchResponse)
async def retrieve(
    request: SearchRequest,
    embedder: QueryEmbedder = Depends(get_embedder),
    client: QdrantClient = Depends(get_qdrant_client),
):
    """
    Retrieve top-k chunks for a given query using hybrid RRF search.
    """
    return await handle_retrieve(
        query=request.query,
        top_k=request.top_k,
        paper_id=request.paper_id,
        has_picture=request.has_picture,
        embedder=embedder,
        client=client,
    )
