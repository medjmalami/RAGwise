"""Local arXiv display metadata keyed by paper ID."""

import json
import re
from functools import lru_cache
from pathlib import Path


def _base_id(paper_id: str) -> str:
    return re.sub(r"v\d+$", "", paper_id)


@lru_cache(maxsize=1)
def _metadata_by_id() -> dict[str, dict]:
    path = Path(__file__).resolve().parents[2] / "arxiv_rag_metadata.jsonl"
    metadata: dict[str, dict] = {}
    try:
        with path.open(encoding="utf-8") as records:
            for line in records:
                record = json.loads(line)
                paper_id = record.get("id")
                if paper_id:
                    metadata[_base_id(str(paper_id))] = record
    except (OSError, json.JSONDecodeError):
        return {}
    return metadata


def arxiv_display_metadata(paper_id: str | None) -> dict[str, str | list[str]]:
    if not paper_id:
        return {}
    record = _metadata_by_id().get(_base_id(paper_id))
    if not record:
        return {}
    published = str(record.get("published") or "")
    result: dict[str, str | list[str]] = {}
    if record.get("title"):
        result["title"] = str(record["title"])
    authors = record.get("authors")
    if authors:
        result["authors"] = authors if isinstance(authors, list) else [str(authors)]
    if published:
        result["year"] = published[:4]
    return result
