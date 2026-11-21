# Repository Guidelines

## Project Structure & Module Organization

- `backend/app/` contains the FastAPI application: authentication, conversations, database models, services, and the RAG graph in `graph/rag_graph.py`.
- `backend/indexing/` contains the arXiv HTML ingestion, preprocessing, parsing, chunking, and embedding pipeline. `backend/app/ragas/` and `backend/app/deepeval/` contain evaluation utilities.
- `backend/alembic/` holds PostgreSQL migrations. `frontend/app/` and `frontend/components/` contain the Next.js chat UI; shared browser API helpers are in `frontend/lib/`.
- `docker-compose.yml` starts local PostgreSQL and Qdrant. Generated corpus files, vector storage, and database data are runtime artifacts, not source files.

## Build, Test, and Development Commands

Run backend commands from `backend/`, where its `.env` is loaded:

```bash
docker compose up -d postgres qdrant
cd backend && uv sync && uv run alembic upgrade head
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Run the frontend from `frontend/` with `bun install` and `bun run dev -- --port 3001`; use `bun run build` to verify a production build. Ollama must be running separately, and backend credentials for Langfuse, Cohere, and the configured model services are required. See `README.md` for indexing and evaluation commands.

## Coding Style & Naming Conventions

Use four-space indentation and `snake_case` for Python modules, functions, and variables. Use two-space indentation in TypeScript/TSX, `camelCase` for functions and variables, and `PascalCase` for React components and types. Match nearby code; no repository-wide formatter or linter is configured.

## Testing Guidelines

There is currently no tracked automated test suite or coverage requirement. For backend changes, apply migrations and exercise affected API routes or the available scripts under `backend/scripts/`; for frontend changes, run `bun run build` and manually check the affected flow. Evaluation scripts are environment- and service-dependent rather than unit tests.

## Commits & Pull Requests

Recent commits use concise imperative subjects, often with prefixes such as `fix:`, `feat:`, and `refactor:` (for example, `fix: make indexing pipeline paths reproducible`). Keep commits focused. Pull requests should explain the behavior change, list validation commands, link related issues, call out migrations or configuration changes, and include UI screenshots when the frontend changes.

## Security & Configuration

Keep secrets in `.env` files and never commit credentials, database contents, Qdrant storage, or generated indexing output. Review changes to authentication, cookies, CORS, and migrations carefully, and document any new environment variables in `README.md`.
