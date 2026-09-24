import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import ChatOllama
from langfuse import Langfuse, get_client
from qdrant_client import QdrantClient

from app.config import settings
from app.graph.rag_graph import build_rag_graph
from app.routes.routes import router
from app.services.rerank import CohereReranker
from app.services.retrieve_from_qdrant import QueryEmbedder


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print("Initializing Langfuse...")
    Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
    )
    print("Starting up: Loading BGE-M3 model...")
    app.state.embedder = QueryEmbedder()
    app.state.embedder.load()

    print("Connecting to Qdrant...")
    app.state.qdrant_client = QdrantClient(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY"),
        prefer_grpc=True,
        grpc_port=6334,
    )

    print("Initializing Gemini LLM...")

    app.state.llm = ChatOllama(
        model="gemma4:31b-cloud",
        temperature=0.2,
    )

    print("Initializing Cohere reranker...")
    app.state.reranker = CohereReranker(
        api_key=settings.cohere_api_key,
        model=settings.cohere_rerank_model,
    )

    print("Building RAG graph...")
    app.state.rag_graph = build_rag_graph(
        embedder=app.state.embedder,
        client=app.state.qdrant_client,
        llm=app.state.llm,
        reranker=app.state.reranker,
    )

    yield  # Server runs here, handling requests

    # --- SHUTDOWN ---
    print("Shutting down: Closing Qdrant connection...")
    app.state.qdrant_client.close()
    get_client().shutdown()


app = FastAPI(lifespan=lifespan)

# Register routes
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
