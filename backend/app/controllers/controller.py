from typing import Optional
from uuid import UUID, uuid4

from fastapi import HTTPException, Request
from langfuse import propagate_attributes
from langfuse.langchain import CallbackHandler
from langgraph.graph.state import CompiledStateGraph

from app.graph.rag_graph import GenerationError, RetrievalError


def get_rag_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.rag_graph


DEFAULT_TOP_K = 5


async def handle_rag_answer(
    query: str,
    conversation_id: Optional[UUID],
    graph: CompiledStateGraph,
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    conversation_id = conversation_id or uuid4()
    handler = CallbackHandler()
    try:
        with propagate_attributes(
            trace_name="rag-answer",
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

    return {
        "conversation_id": conversation_id,
        "query": query,
        "answer": result["answer"],
        "sources": result.get("chunks", []),
    }
