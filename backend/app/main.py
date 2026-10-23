import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import ChatOllama
from langfuse import Langfuse, get_client
from qdrant_client import QdrantClient

from app.auth.routes.auth_routes import router as auth_router
from app.config.config import settings
from app.config.telemetry import create_tracer_provider
from app.graph.rag_graph import build_rag_graph
from app.routes.routes import router
from app.services.rerank import CohereReranker
from app.services.retrieve_from_qdrant import QueryEmbedder
from app.services.vision_llm import PathImageChatOllama


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

    print("Initializing Ollama LLM...")

    app.state.llm = PathImageChatOllama(
        model="gemma4:31b-cloud",
        temperature=0.2,
        num_ctx=16384,  # eval sets this, main.py doesn't
    )

    print("Initializing summary LLM...")
    app.state.summary_llm = ChatOllama(
        model="gemma4:31b-cloud",
        temperature=0,
        num_ctx=8192,
        num_predict=512,
        reasoning=False,
    )

    print("Initializing title LLM...")
    app.state.title_llm = ChatOllama(
        model="gemma4:31b-cloud",
        temperature=0.3,
        num_ctx=2048,
        num_predict=64,
        reasoning=False,
    )

    print("Initializing rewrite LLM...")
    app.state.rewrite_llm = ChatOllama(
        model="gemma4:31b-cloud",  # swap for a smaller model if you have one
        temperature=0,
        num_ctx=4096,
        num_predict=128,
        reasoning=False,
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
        rewrite_llm=app.state.rewrite_llm,
    )

    yield  # Server runs here, handling requests

    # --- SHUTDOWN ---
    print("Shutting down: Closing Qdrant connection...")
    app.state.qdrant_client.close()
    get_client().shutdown()


app = FastAPI(
    lifespan=lifespan,
    telemetry={"tracer_provider": create_tracer_provider(), "auto_configure": False},
)

# Register routes
app.include_router(router)
app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
