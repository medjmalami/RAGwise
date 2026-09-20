from typing import Optional

from fastapi import HTTPException, Request
from langfuse import propagate_attributes
from langfuse.langchain import CallbackHandler
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

    handler = CallbackHandler()
    try:
        with propagate_attributes(
            trace_name="rag-answer",
            tags=["rag"],
            metadata={
                "top_k": str(top_k),
                "paper_id": paper_id or "none",
                "has_picture": str(has_picture),
            },
        ):
            result = await graph.ainvoke(
                {
                    "query": query,
                    "top_k": top_k,
                    "paper_id": paper_id,
                    "has_picture": has_picture,
                },
                config={"callbacks": [handler]},
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
