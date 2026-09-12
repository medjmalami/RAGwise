from typing import Optional

from fastapi import HTTPException, Request
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient, models

from app.services.retrieve_from_qdrant import (
    DEFAULT_COLLECTION,
    QueryEmbedder,
    retrieve_top_k,
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
def get_embedder(request: Request) -> QueryEmbedder:
    return request.app.state.embedder


def get_qdrant_client(request: Request) -> QdrantClient:
    return request.app.state.qdrant_client


def get_llm(request: Request) -> ChatGoogleGenerativeAI:
    return request.app.state.llm


# ---------------------------------------------------------------------------
# Controller Logic
# ---------------------------------------------------------------------------
async def handle_rag_answer(
    query: str,
    top_k: int,
    paper_id: Optional[str],
    has_picture: Optional[bool],
    embedder: QueryEmbedder,
    client: QdrantClient,
    llm: ChatGoogleGenerativeAI,
):
    """Retrieves chunks and generates an answer using Gemini."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # 1. Build Qdrant filter dynamically based on request
    must_conditions = []
    if paper_id:
        must_conditions.append(
            models.FieldCondition(
                key="paper_id", match=models.MatchValue(value=paper_id)
            )
        )
    if has_picture is not None:
        must_conditions.append(
            models.FieldCondition(
                key="has_picture", match=models.MatchValue(value=has_picture)
            )
        )
    query_filter = models.Filter(must=must_conditions) if must_conditions else None

    # 2. Retrieve chunks
    try:
        chunks = retrieve_top_k(
            query=query,
            client=client,
            embedder=embedder,
            collection=DEFAULT_COLLECTION,
            k=top_k,
            query_filter=query_filter,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    if not chunks:
        return {
            "query": query,
            "answer": "I couldn't find any relevant documents to answer your question.",
            "sources": [],
        }

    # 3. Format the chunks into a single context string
    context_text = "\n\n---\n\n".join(
        [f"Paper: {c['paper_id']}\nContent: {c['text']}" for c in chunks]
    )

    # 4. Create the LangChain Prompt Template
    prompt = ChatPromptTemplate.from_template("""
    You are an expert AI assistant. Use the following pieces of retrieved context
    to answer the user's question. If you don't know the answer based on the context,
    just say you don't know. Don't try to make up an answer.

    Context:
    {context}

    Question: {question}

    Answer:
    """)

    # 5. Build the LCEL chain and invoke it
    chain = prompt | llm | StrOutputParser()

    try:
        answer = chain.invoke({"context": context_text, "question": query})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    # 6. Return the answer and the source chunks (for UI citations)
    return {"query": query, "answer": answer, "sources": chunks}
