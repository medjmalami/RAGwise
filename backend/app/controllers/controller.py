from typing import Optional

from fastapi import HTTPException, Request
from qdrant_client import QdrantClient, models

from app.services.retrieve_from_qdrant import (
    DEFAULT_COLLECTION,
    QueryEmbedder,
    retrieve_top_k,
)


# ---------------------------------------------------------------------------
# Dependencies: Access shared state loaded in main.py
# ---------------------------------------------------------------------------
def get_embedder(request: Request) -> QueryEmbedder:
    """Retrieves the BGE-M3 model loaded during app startup."""
    return request.app.state.embedder


def get_qdrant_client(request: Request) -> QdrantClient:
    """Retrieves the Qdrant client loaded during app startup."""
    return request.app.state.qdrant_client


# ---------------------------------------------------------------------------
# Controller Logic
# ---------------------------------------------------------------------------
async def handle_retrieve(
    query: str,
    top_k: int,
    paper_id: Optional[str],
    has_picture: Optional[bool],
    embedder: QueryEmbedder,
    client: QdrantClient,
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Build Qdrant filter dynamically based on request
    must_conditions = []
    if paper_id:
        must_conditions.append(
            models.FieldCondition(
                key="paper_id",
                match=models.MatchValue(value=paper_id),
            )
        )
    if has_picture is not None:
        must_conditions.append(
            models.FieldCondition(
                key="has_picture",
                match=models.MatchValue(value=has_picture),
            )
        )

    query_filter = models.Filter(must=must_conditions) if must_conditions else None

    try:
        hits = retrieve_top_k(
            query=query,
            client=client,
            embedder=embedder,
            collection=DEFAULT_COLLECTION,
            k=top_k,
            query_filter=query_filter,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    # Format response
    return {"query": query, "results": hits}
