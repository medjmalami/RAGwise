"""Canonical local paths shared by the indexing stages.

Keeping these paths next to the indexing code makes the pipeline independent
of the directory from which a stage is launched.  The generated data remains
under ``backend/arxiv_rag_papers`` so it also matches the application image
loader and the commands documented in the repository README.
"""

from pathlib import Path


INDEXING_DIR = Path(__file__).resolve().parent
BACKEND_DIR = INDEXING_DIR.parent

METADATA_FILE = BACKEND_DIR / "arxiv_rag_metadata.jsonl"
PAPERS_DIR = BACKEND_DIR / "arxiv_rag_papers"

HTML_DIR = PAPERS_DIR / "html"
ASSETS_DIR = PAPERS_DIR / "assets"
PREPROCESSED_HTML_DIR = PAPERS_DIR / "preprocessed_html"
PARSED_DIR = PAPERS_DIR / "parsed"
CHUNKS_DIR = PAPERS_DIR / "chunks"

FAILED_FILE = PAPERS_DIR / "html_failed.jsonl"
IMAGE_FAILED_FILE = PAPERS_DIR / "image_failed.jsonl"
EMBED_STATE_DIR = BACKEND_DIR / "embed_state"
