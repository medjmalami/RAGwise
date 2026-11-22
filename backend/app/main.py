import asyncio
import logging
import math
import os
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langfuse import Langfuse, get_client
from qdrant_client import QdrantClient

from app.auth.routes.auth_routes import router as auth_router
from app.config.config import settings
from app.config.telemetry import create_tracer_provider
from app.conversations.routes.conversation_routes import router as conversations_router
from app.graph.rag_graph import build_rag_graph
from app.routes.routes import router
from app.services.llm_factory import build_llms
from app.services.semantic_cache import SemanticCache
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

    print("Initializing LLMs...")
    llms = build_llms()
    app.state.llm = llms.main
    app.state.summary_llm = llms.summary
    app.state.title_llm = llms.title
    app.state.rewrite_llm = llms.rewrite

    print("Initializing Cohere reranker...")
    app.state.reranker = CohereReranker(
        api_key=settings.cohere_api_key,
        model=settings.cohere_rerank_model,
    )

    cache = None
    cache_client = None
    cleanup_task = None
    if settings.semantic_cache_enabled:
        # A separate transport bounds cache I/O without shortening retrieval timeouts.
        try:
            cache_client = QdrantClient(
                url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
                api_key=os.environ.get("QDRANT_API_KEY"),
                timeout=max(1, math.ceil(settings.semantic_cache_timeout_seconds)),
            )
            candidate_cache = SemanticCache(cache_client, settings.cache_options())
            await candidate_cache.initialize()
            cache = candidate_cache
            cleanup_task = asyncio.create_task(cache.cleanup_loop())
        except Exception:
            logging.getLogger(__name__).exception(
                "Cache initialization failed; serving without cache"
            )

    print("Building RAG graph...")
    app.state.rag_graph = build_rag_graph(
        embedder=app.state.embedder,
        client=app.state.qdrant_client,
        llm=app.state.llm,
        reranker=app.state.reranker,
        rewrite_llm=app.state.rewrite_llm,
        use_cache=cache is not None,
        cache=cache,
    )

    try:
        yield  # Server runs here, handling requests
    finally:
        # --- SHUTDOWN ---
        print("Shutting down: Closing Qdrant connection...")
        if cleanup_task is not None:
            cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup_task
        if cache_client is not None:
            cache_client.close()
        app.state.qdrant_client.close()
        get_client().shutdown()


app = FastAPI(
    lifespan=lifespan,
    telemetry={"tracer_provider": create_tracer_provider(), "auto_configure": False},
)

# Register routes
app.include_router(router)
app.include_router(auth_router)
app.include_router(conversations_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
