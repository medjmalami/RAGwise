# RAGWise

RAGWise is a research-paper question-answering application. It indexes arXiv papers, retrieves passages from Qdrant, and answers questions with an Ollama vision-capable model. The repository contains a FastAPI backend, a Next.js chat interface, indexing scripts, and a RAGAS evaluation harness.

The checked-in `backend/arxiv_rag_metadata.jsonl` contains 3,069 arXiv metadata records. The current indexing flow fetches arXiv HTML from `arxiv.org/html`, records unavailable papers, and parses HTML; it does not use an ar5iv fallback or process PDFs. The separate `papersDownload.py` utility can download PDFs, but it is not part of this parsing flow.

## How it works

```text
arXiv HTML and images
  → preprocessing and Docling parsing
  → token-aware chunks with paper and image metadata
  → BGE-M3 dense and sparse embeddings in Qdrant
  → conversation-aware query rewrite
  → hybrid retrieval and optional Cohere reranking
  → Ollama answer generation, with relevant figures when available
```

The backend streams answers and retrieved sources to the UI as Server-Sent Events. Accounts, refresh sessions, conversations, messages, and conversation summaries are stored in PostgreSQL. Langfuse receives traces; RAGAS scripts can evaluate a Langfuse dataset.

## Repository layout

| Path | Purpose |
|---|---|
| `backend/app/main.py` | FastAPI application and startup initialization |
| `backend/app/auth/` | Sign-up, sign-in, refresh, logout, and current-user routes |
| `backend/app/conversations/` | Conversation and message history endpoints |
| `backend/app/graph/rag_graph.py` | Query rewriting, retrieval, reranking, image selection, and generation |
| `backend/app/services/` | Qdrant retrieval, Cohere reranking, conversation summaries, and LLM helpers |
| `backend/indexing/` | arXiv acquisition, preprocessing, parsing, chunking, and embedding scripts |
| `backend/app/ragas/` | Dataset preparation and RAGAS evaluation utilities |
| `backend/alembic/` | PostgreSQL schema migrations |
| `frontend/` | Next.js chat and account UI |
| `docker-compose.yml` | Local PostgreSQL and Qdrant services |

The HTML, assets, parsed files, chunk files, embedding markers, Qdrant data, and secrets are local runtime data and are not tracked. The metadata JSONL and a testset CSV are checked in.

## Requirements

- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/)
- Node.js 20 or newer and Bun (or npm)
- Docker with Compose
- An Ollama server accessible at `http://localhost:11434`, with `gemma4:31b-cloud` available to it
- API credentials for Langfuse and Cohere. `GEMINI_API_KEY` is also a required backend setting and is used for picture descriptions when indexing enables them.

The backend loads BGE-M3 at startup. The initial run downloads model files and can take time; GPU use is automatic when CUDA is available, otherwise the embedder uses CPU.

## Local setup

### 1. Configure environment

At the repository root, copy the Compose settings template:

```bash
cp .env.example .env
```

Set `DB_USER`, `DB_PASSWORD`, and `DB_NAME` for PostgreSQL. The root `.env` is read by Docker Compose.

Create `backend/.env` from `backend/.env.example` and provide the backend settings below. The backend reads `.env` from its current working directory, so run backend commands from `backend/`.

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | SQLAlchemy async URL, for example `postgresql+asyncpg://postgres:postgres@localhost:5432/ragwise` |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | Langfuse tracing configuration |
| `COHERE_API_KEY` | Cohere reranking |
| `OTEL_SERVICE_NAME` | OpenTelemetry service name |
| `JWT_SECRET` | Secret used to sign access tokens |
| `GEMINI_API_KEY` | Required by the backend configuration and for picture descriptions during parsing |
| `COHERE_RERANK_MODEL` | Optional; defaults to `rerank-v3.5` |
| `COOKIE_SECURE` | Optional; defaults to `true`. Set to `false` for local HTTP sign-in cookies. |

`QDRANT_URL` and `QDRANT_API_KEY` are optional; the defaults are `http://localhost:6333` and no API key. History-summary thresholds and token settings also have defaults in `backend/app/config/config.py`.

The frontend uses `NEXT_PUBLIC_API_URL`, defaulting to `http://localhost:8000`. No frontend env file is required for the local defaults.

### 2. Start PostgreSQL and Qdrant

From the repository root:

```bash
docker compose up -d postgres qdrant
```

### 3. Install backend dependencies and migrate the database

```bash
cd backend
uv sync
uv run alembic upgrade head
```

### 4. Start Ollama and the API

Make sure Ollama is running and can serve `gemma4:31b-cloud`. Then, from `backend/`:

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API initializes the BGE-M3 embedder and clients for Qdrant, Langfuse, Cohere, and Ollama. Interactive API documentation is available at `http://localhost:8000/docs`.

### 5. Start the web app

In a second terminal, from the repository root:

```bash
cd frontend
bun install
bun run dev -- --port 3001
```

Open `http://localhost:3001`. The backend CORS configuration currently allows this local origin. With npm, use `npm install` and `npm run dev -- --port 3001`.

## Using the application

Create an account or sign in, then ask a question about the indexed research papers. Answers stream into the chat with a sources panel showing retrieved passages and arXiv links. Conversations are saved to the signed-in user's account and can be reopened from the sidebar.

The API exposes:

| Route | Description |
|---|---|
| `POST /auth/signup`, `POST /auth/signin` | Create an account or sign in; returns a short-lived access token and sets an HTTP-only refresh cookie |
| `POST /auth/refresh`, `POST /auth/logout`, `GET /auth/me` | Refresh or end a session and inspect the current account |
| `POST /answer` | Authenticated question endpoint; returns an SSE stream with title, sources, token, reset, done, and error events |
| `GET /conversations` | List the user's conversations with cursor pagination |
| `GET /conversations/{conversation_id}` | Read a conversation and its messages with pagination |

`POST /answer` accepts `{"query":"..."}` to start a conversation or `{"query":"...","conversation_id":"..."}` to continue one. Its final `done` event contains the saved answer and conversation ID.

### Conversation context

The full conversation remains in PostgreSQL, while the model receives recent turns verbatim and a rolling summary of older turns when the estimated context exceeds the trigger. The conversation watermark `summary_upto_message_id` records which message was last folded into that summary.

| Setting | Default | Purpose |
|---|---:|---|
| `HISTORY_TRIGGER_TOKENS` | `5000` | Estimated context size that triggers compaction |
| `HISTORY_TARGET_TOKENS` | `2500` | Target size after compacting older turns |
| `HISTORY_MIN_KEEP_TURNS` | `3` | Minimum recent turns kept verbatim |

Token counts are estimates based on message characters and overhead, not counts from the Ollama tokenizer.

## Indexing a paper collection

The indexing pipeline is optional if you already have a populated Qdrant collection. Its local working directories live under `backend/arxiv_rag_papers/` and are ignored by Git.

Run the commands from `backend/`:

```bash
# Download available arXiv HTML and image assets into arxiv_rag_papers/
uv run python -m indexing.ingestion

# Clean arXiv page chrome and inline locally downloaded images
uv run python -m indexing.preprocess

# Parse preprocessed HTML into Docling documents (needs GEMINI_API_KEY unless disabled)
uv run python -m indexing.parser \
  --input arxiv_rag_papers/preprocessed_html \
  --output arxiv_rag_papers/parsed

# Create contextualized, token-aware chunks
uv run python -m indexing.chunker \
  --input arxiv_rag_papers/parsed \
  --output arxiv_rag_papers/chunks \
  --tokenizer BAAI/bge-m3 \
  --max-tokens 512

# Embed chunks and upsert dense and sparse vectors to Qdrant
uv run python -m indexing.embeder \
  --input arxiv_rag_papers/chunks \
  --collection ragwise_arxiv
```

The ingestion and preprocessing scripts use paths relative to their own source directory. Parser, chunker, and embedder accept `--limit` for small runs. The embedder also accepts `--device cpu` or `--device cuda`; see each script's `--help` for resumability, batch, and collection options. Parsing with `--no-picture-description` avoids the Gemini request, but figure descriptions will not be added.

The API and evaluation harness query the `ragwise_arxiv` collection by default. Keep the collection name and embedding model consistent between indexing and serving.

## Evaluation

`backend/app/ragas/evaluate_rag.py` evaluates a Langfuse dataset with the RAG graph and RAGAS metrics. It uses Ollama as the generator and judge, BGE-M3 embeddings for answer relevancy, and records the run and scores in Langfuse. The default dataset name is `ragwise_ragas_testset`; override it with `--dataset`.

Create and upload a testset in this order. Run the commands from `backend/` with the required backend settings configured as above:

1. Build the knowledge graph from Qdrant chunks:

   ```bash
   uv run python -m app.ragas.generate_knowledge_graph
   ```

   This script reads up to 1,000 records from the local `ragwise_arxiv` collection, samples up to 200, and skips records without text. It applies RAGAS default transforms, saves `knowledge_graph.json`, and also generates a 10-item `ragas_test_set.csv`.

2. Generate the evaluation CSV from that graph. This command requests 100 items; generation can return fewer if some samples fail:

   ```bash
   uv run python -m app.ragas.generate_testset \
     --kg-path knowledge_graph.json \
     --output-csv testset.csv \
     --testset-size 100
   ```

   The script's options and defaults are `--kg-path` (`knowledge_graph.json`), `--output-csv` (`testset.csv`), `--testset-size` (`10`), `--model` (`gemma4:31b-cloud`), `--ollama-base-url` (`http://localhost:11434`), `--embedding-model` (`BAAI/bge-m3`), `--max-workers` (`3`), `--max-wait` (`90` seconds), and `--max-retries` (`8`). The graph-building stage uses the local Qdrant collection, BGE-M3 embeddings, and an Ollama `gemma4:31b-cloud` model; it also sets the configured `GEMINI_API_KEY` as `GOOGLE_API_KEY`. The generation stage uses Ollama and BGE-M3. Both scripts import the backend settings, so all required backend environment settings must be present.

   The CSV contains generated questions (`user_input`), expected answers (`reference`), and source contexts (`reference_contexts`). The graph builder selects at most 200 chunks; missing or textless payloads can reduce the node count, and relation counts depend on the transforms. The checked-in code does not fix either graph count.

3. Upload the CSV to Langfuse:

   ```bash
   uv run python -m app.ragas.upload_testset_langfuse \
     testset.csv \
     --dataset-name ragwise_ragas_testset
   ```

   The uploader takes a positional CSV path and supports `--dataset-name` (default `ragwise_ragas_testset`) and `--description` (default `RAGAS-generated testset from arXiv chunk sample`). It requires the Langfuse public key, secret key, and base URL, and checks authentication before creating dataset items.

Then evaluate the uploaded dataset:

```bash
cd backend
uv run python -m app.ragas.evaluate_rag --limit 1
uv run python -m app.ragas.evaluate_rag --run-name baseline --top-k 5
```

Options include `--dataset` (default `ragwise_ragas_testset`), `--run-name`, `--top-k`, `--limit`, `--concurrency`, and `--no-images`. The Ollama generator and judge model names, base URL, context length, and embedding device can be overridden with `OLLAMA_GENERATOR_MODEL`, `OLLAMA_JUDGE_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_NUM_CTX`, and `EMBED_DEVICE` (set `EMBED_DEVICE=cpu` when evaluating without CUDA).

### Evaluation results

These results were measured on the 100-question synthetic testset with RAGAS, using local Ollama `gemma4:31b` as the judge and BGE-M3 embeddings for answer relevancy.

| Variant | Avg latency | Answer relevancy | Context precision | Context recall | Faithfulness |
|---|---:|---:|---:|---:|---:|
| Hybrid dense+sparse baseline | 1.6538s | 0.71 | 0.45 | 0.67 | 0.92 |
| + Cohere rerank | 3.2212s | 0.71 | 0.60 | 0.72 | 0.96 |
| + Multimodal figures | 3.8487s | 0.75 | 0.60 | 0.72 | 0.96 |

Cohere reranking raised context precision by about 33%, context recall by about 7%, and faithfulness by about 4%; answer relevancy moved from 0.73 to 0.71, while latency was roughly 2x the hybrid baseline. The multimodal run used figures for 16 of 100 questions. Compared with reranking, its aggregate answer relevancy was 0.75, precision 0.59, recall 0.72, and faithfulness 0.95, with average latency of 3.8487s; on some image-triggered questions, answer relevancy rose while faithfulness fell.

These results come from a single run on a 100-question synthetic testset, so differences of a few points are within noise. Query rewriting was not benchmarked because this testset is single-turn and does not support multi-turn evaluation; that requires a separate testset.

## Development notes

- Run Python commands with `backend/` as the working directory; the backend package and its `.env` are resolved from there.
- The Compose file starts PostgreSQL and Qdrant only. Run Ollama separately. Langfuse can run as a local self-hosted v4 instance or through Langfuse Cloud; `LANGFUSE_BASE_URL` selects the endpoint. See the [official Langfuse self-hosting documentation](https://langfuse.com/self-hosting) for deployment instructions.
- Langfuse configuration is required: the backend settings require its keys and base URL at import, and startup initializes the Langfuse client and trace exporter. The application does not provide a no-Langfuse mode.
- Database migrations are managed by Alembic. Apply them with `uv run alembic upgrade head` from `backend/`.
- The chat frontend is configured for the local backend at port 8000 and local CORS origin at port 3001. If you change either port, update the frontend URL or backend CORS settings accordingly.
