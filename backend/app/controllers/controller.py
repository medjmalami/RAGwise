from typing import Optional

from fastapi import HTTPException, Request
from langgraph.graph.state import CompiledStateGraph

from app.graph.rag_graph import GenerationError, RetrievalError


def get_rag_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.rag_graph


async def handle_rag_answer(
    query: str,
    top_k: int,
    paper_id: Optional[str],
    has_picture: Optional[bool],
    graph: CompiledStateGraph,
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = await graph.ainvoke(
            {
                "query": query,
                "top_k": top_k,
                "paper_id": paper_id,
                "has_picture": has_picture,
            }
        )
    except RetrievalError as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")
    except GenerationError as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

    return {
        "query": query,
        "answer": result["answer"],
        "sources": result.get("chunks", []),
    }
