import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient

from app.config import settings
from app.routes.routes import router
from app.services.retrieve_from_qdrant import QueryEmbedder


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
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
    # 3. Use GEMINI_API_KEY instead of GOOGLE_API_KEY
    app.state.llm = ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        temperature=0.2,
        google_api_key=settings.gemini_api_key,
    )

    yield  # Server runs here, handling requests

    # --- SHUTDOWN ---
    print("Shutting down: Closing Qdrant connection...")
    app.state.qdrant_client.close()


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
