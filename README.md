# RAGwise – Retrieval-Augmented Generation over ~3k arXiv HTML Papers

RAGwise is a complete end-to-end Retrieval-Augmented Generation (RAG) pipeline that ingests ~3,069 arXiv papers (HTML primary, ar5iv fallback), parses them into structured Docling documents with picture-description enrichment, chunks them with a token-aware `HybridChunker`, embeds each chunk with BGE-M3 (dense + sparse), stores the vectors in Qdrant, and serves answers via a LangGraph-driven FastAPI backend backed by an Ollama LLM.

PDF ingestion is deliberately omitted — the available hardware can't process large PDFs efficiently, so arXiv HTML (with ar5iv as fallback) is the primary input format.

The repo also ships an evaluation harness that runs the graph against a Langfuse test set, scores the results with RAGAS, and writes the metrics back to Langfuse.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Tech Stack](#tech-stack)
3. [Setup & Installation](#setup--installation)
4. [Running the Pipeline](#running-the-pipeline)
5. [Evaluation](#evaluation)
6. [Roadmap (Not Yet Done)](#roadmap-not-yet-done)
7. [Project Structure](#project-structure)

## Architecture Overview

```
HTML papers (arXiv) ──► parser.py ──► DoclingDocument + Markdown
                         │
                         └─► picture-description (Gemini/Gemma)
                               │
                               ▼
               chunker.py ──► HybridChunker (tokenizer = BGE-M3)
                               │
                               ▼
                embedder.py ──► BGEM3FlagModel (dense + lexical)
                               │
                               ▼
                     Qdrant collection (dense & sparse vectors)
                               │
                               ▼
        retrieve_from_qdrant.py (hybrid search + RRF fusion)
                               │
                               ▼
            rerank.py (Cohere, optional)
                               │
                               ▼
          LangGraph rag_graph (retrieve → generate)
                               │
                               ▼
          LLM via Ollama (gemma4:31b-cloud) → answer
```

- **Stage 1 – Parse**: `backend/indexing/parser.py` converts arXiv HTML → Docling Document & Markdown, calling the Gemini/Gemma picture-description API when enabled.
- **Stage 2 – Chunk**: `backend/indexing/chunker.py` uses `HybridChunker` bound to the BGE-M3 tokenizer (max tokens configurable).
- **Stage 3 – Embed**: `backend/indexing/embeder.py` runs `BAAI/bge-m3` (dense + sparse) and upserts each chunk into a Qdrant collection (`ragwise_arxiv`).
- **Retrieve**: `backend/app/services/retrieve_from_qdrant.py` performs hybrid search (dense cosine + BM25-style lexical) and fuses results with Reciprocal Rank Fusion (RRF).
- **Rerank (optional)**: `backend/app/services/rerank.py` calls the Cohere reranker on the top-k hybrid candidates.
- **Generate**: `backend/app/graph/rag_graph.py` defines a sequential LangGraph (`START → retrieve → generate → END`). The generation node invokes an Ollama LLM (`gemma4:31b-cloud`).

All components are wired together in a FastAPI server (`backend/app/main.py`) exposing the `/answer` endpoint.

## Tech Stack

| Layer | Tool / Library | Role |
|---|---|---|
| Package manager | uv | Installs and isolates Python dependencies (`uv sync`) |
| HTML parsing | Docling (HTML backend) | Turns arXiv HTML into Docling Document + Markdown |
| Picture description | Gemini / Gemma-4 (OpenAI-compatible endpoint) | Generates 2–3 sentence figure captions at parse time |
| Chunking | HybridChunker (docling_core) | Token-aware chunking aligned to the BGE-M3 tokenizer |
| Embeddings | BAAI/bge-m3 (FlagEmbedding) | Dense (1024-dim) + sparse (lexical) vectors per chunk |
| Vector store | Qdrant | Stores dense & sparse vectors + payload (`paper_id`, `has_picture`, etc.) |
| Retrieval | Custom helper (`retrieve_from_qdrant.py`) | Hybrid search with RRF fusion; configurable prefetch limits |
| Reranking | Cohere (`rerank.py`, model `rerank-v3.5`) | Optional top-k rerank after hybrid retrieval |
| LLM generation | Ollama (`gemma4:31b-cloud`) | Answers user queries |
| Orchestration | LangGraph (`rag_graph.py`) | Sequential graph: retrieve → generate |
| Web API | FastAPI (`backend/app/main.py`) | Exposes `/answer` endpoint |
| Frontend | Next.js (React) | UI for query input and answer display |
| Eval stack | Langfuse, RAGAS | End-to-end evaluation and monitoring |
| DB migrations | Alembic | Manages schema for Langfuse tables |

## Setup & Installation

### Prerequisites

| Software | Version / Notes |
|---|---|
| Python | 3.11 (managed by uv) |
| Node | 20+ |
| Docker | For Qdrant & PostgreSQL (optional) |
| Ollama | Running locally (`ollama serve`) with `gemma4:31b-cloud` pulled |
| API keys | `GEMINI_API_KEY`, `COHERE_API_KEY`, Langfuse keys |

### 1. Clone the repository

```bash
git clone https://github.com/yourorg/RAGwise.git
cd RAGwise
```

> TODO: replace with the actual repository URL.

### 2. Environment variables

```bash
cp .env.example .env
cp frontend/.env.example frontend/.env
```

Fill in `.env` (backend):

```
DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/ragwise
GEMINI_API_KEY=YOUR_GEMINI_KEY
COHERE_API_KEY=YOUR_COHERE_KEY
COHERE_RERANK_MODEL=rerank-v3.5
LANGFUSE_PUBLIC_KEY=YOUR_PUBLIC_KEY
LANGFUSE_SECRET_KEY=YOUR_SECRET_KEY
LANGFUSE_BASE_URL=https://api.langfuse.com
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_JUDGE_MODEL=gemma4:31b-cloud
OLLAMA_GENERATOR_MODEL=gemma4:31b-cloud
EMBED_MODEL=BAAI/bge-m3
EMBED_DEVICE=cuda   # or cpu
```

### 3. Install Python dependencies

```bash
uv sync                      # creates .venv and installs everything from pyproject.toml
source .venv/bin/activate    # optional; uv already configures PATH
```

### 4. Install frontend dependencies

```bash
cd frontend
bun install    # or `npm install`
```

### 5. Start supporting services

```bash
# Qdrant (vector DB)
docker compose up -d qdrant

# PostgreSQL (Langfuse & Alembic migrations)
docker compose up -d postgres

# Ollama server (separate terminal)
ollama serve
ollama pull gemma4:31b-cloud
```

### 6. Run DB migrations (if Langfuse tables are needed)

```bash
alembic upgrade head
```

## Running the Pipeline

> The repository already ships pre-processed `parsed/`, `chunks/`, and a populated Qdrant collection under `qdrant_storage/`. Skip the data-generation steps below unless you want to re-run them.

### 1. Optional: re-process the raw HTML corpus

```bash
# Stage 1 – Parse HTML → DoclingDocument (+ picture captions)
python -m backend.indexing.parser \
    --input arxiv_html/ \
    --output parsed/ \
    --gemma-model gemma-4-31b-it

# Stage 2 – Chunk
python -m backend.indexing.chunker \
    --input parsed/ \
    --output chunks/ \
    --tokenizer BAAI/bge-m3 \
    --max-tokens 512

# Stage 3 – Embed & upsert to Qdrant
python -m backend.indexing.embeder \
    --input chunks/ \
    --collection ragwise_arxiv \
    --device cuda   # omit for auto-detect
```

### 2. Start the FastAPI backend

```bash
uv run python -m backend.app.main
# → API listening on http://127.0.0.1:8000
```

### 3. Launch the UI

```bash
cd frontend
bun dev    # or `npm run dev`
# Open http://localhost:3000
```

Enter a natural-language question; the UI calls `/answer`, which runs the LangGraph (retrieve → generate) and returns the answer together with the retrieved chunks.

## Evaluation

Script: `backend/app/ragas/evaluate_rag.py`

```bash
uv run python -m backend.app.ragas.evaluate_rag \
    --limit 100 \
    --run-name v1 \
    --top-k 5
```

| Step | Description |
|---|---|
| Dataset | Loads the Langfuse dataset `ragwise_ragas_testset` |
| Health check | Verifies Ollama is reachable and the required models are present |
| Graph execution | Calls `build_rag_graph` → runs retrieve + generate for each test item |
| RAGAS metrics | Computes Precision, Recall, Faithfulness, Answer Relevancy — relevancy uses BGE-M3 embeddings (local HF cache), not Ollama |
| Judge LLM | Uses the local Ollama model (`gemma4:31b-cloud`) as the RAGAS judge |
| Result storage | Writes each metric back to the corresponding Langfuse trace / dataset run item |

The script aborts early with a clear error if Ollama is down or the required models haven't been pulled.

## Roadmap (Not Yet Done)

- **Full-conversation query rewriting** – use prior dialogue turns to reformulate the current query, instead of just the last message.
- **Advanced LLM-based reranking** – extend beyond Cohere with models such as GPT-4o or a locally hosted Mistral.
- **Agent / skill-based RAG** – dynamic tool-calling inside the generation step, so the system acts as an agent instead of always retrieving.
- **GraphRAG** – integrate a structured knowledge graph with LLM-driven retrieval for richer context.

> Note: hybrid (dense + sparse) retrieval is already the current baseline, not a planned addition.

## Project Structure

```
RAGwise/
├─ backend/
│  ├─ app/
│  │  ├─ config.py                  # Pydantic Settings (env vars)
│  │  ├─ graph/
│  │  │  └─ rag_graph.py            # LangGraph pipeline (retrieve → generate)
│  │  ├─ ragas/
│  │  │  ├─ evaluate_rag.py         # Evaluation script (RAGAS + Langfuse)
│  │  │  └─ …                       # dataset helpers
│  │  ├─ routes/
│  │  │  └─ routes.py               # FastAPI endpoint definitions
│  │  ├─ services/
│  │  │  ├─ retrieve_from_qdrant.py # Hybrid retrieval helpers
│  │  │  └─ rerank.py               # Cohere reranker
│  │  └─ main.py                    # FastAPI entry point
│  ├─ indexing/
│  │  ├─ parser.py                  # Stage 1 – HTML → Docling + Gemini picture desc.
│  │  ├─ chunker.py                 # Stage 2 – HybridChunker
│  │  ├─ embeder.py                 # Stage 3 – BGE-M3 embed & Qdrant upsert
│  │  └─ …                          # utilities (download scripts, manifest, etc.)
│  ├─ alembic/                      # DB migrations for Langfuse tables
│  ├─ qdrant_storage/               # Pre-populated Qdrant collection
│  ├─ testset.csv                   # Snapshot of the Langfuse test set
│  └─ .env.example
├─ frontend/
│  ├─ app/                          # Next.js pages & components
│  ├─ components/
│  ├─ lib/
│  ├─ public/
│  └─ .env.example
├─ docker-compose.yml               # Services: qdrant, postgres, (optional) ollama
├─ .env.example
└─ README.md
```

---

If you modify any pipeline stage, re-run the corresponding stage(s) and restart the API to pick up the changes. Issues and pull requests for new features or fixes are welcome.
