"""Qdrant storage for first-turn responses; callers handle failures fail-open."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from app.services.retrieve_from_qdrant import DENSE_DIM

logger = logging.getLogger(__name__)
DEFAULT_CACHE_COLLECTION = "ragwise_cache"


@dataclass(frozen=True)
class CacheOptions:
    collection: str = DEFAULT_CACHE_COLLECTION
    threshold: float = 0.92
    ttl_seconds: int = 7 * 24 * 60 * 60
    timeout_seconds: float = 2.0
    corpus_version: str = "1"

    def __post_init__(self):
        if (
            not 0 <= self.threshold <= 1
            or self.ttl_seconds <= 0
            or self.timeout_seconds <= 0
        ):
            raise ValueError("Invalid semantic cache threshold, TTL, or timeout")


def fingerprint(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def ensure_cache_collection(client: QdrantClient, name: str) -> bool:
    """Ensure indexes even on existing collections; never recreate stored data."""
    created = not client.collection_exists(name)
    if created:
        client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": models.VectorParams(
                    size=DENSE_DIM, distance=models.Distance.COSINE
                )
            },
        )
    info = client.get_collection(name)
    vectors = info.config.params.vectors
    if (
        not isinstance(vectors, dict)
        or "dense" not in vectors
        or vectors["dense"].size != DENSE_DIM
        or vectors["dense"].distance != models.Distance.COSINE
    ):
        raise ValueError("Incompatible semantic cache vector configuration")
    for field, schema in [
        ("cache_version", models.PayloadSchemaType.KEYWORD),
        ("top_k", models.PayloadSchemaType.INTEGER),
        ("expires_at", models.PayloadSchemaType.INTEGER),
    ]:
        if field not in info.payload_schema:
            client.create_payload_index(
                collection_name=name, field_name=field, field_schema=schema
            )
    return created


def write_skip_reason(state: dict) -> str | None:
    if not state.get("chunks"):
        return "no_chunks"
    if state.get("rerank_fallback") or state.get("image_fallback"):
        return "fallback"
    answer = (state.get("answer") or "").strip()
    if not answer:
        return "empty_answer"
    text = answer.lower().replace("’", "'")
    if re.search(
        r"\bi (?:don'?t|do not) know\b|\b(?:cannot|can't|couldn't|could not|unable to) (?:answer|determine|find|provide|conclude)\b|\b(?:insufficient|not enough) (?:information|context|evidence)\b|\b(?:context|documents?|passages?) (?:does not|do not|doesn't|don't) (?:provide|contain|include)\b",
        text,
    ):
        return "abstention"
    return None


def validate_payload(payload: dict, images_root: Path | None) -> dict:
    answer, chunks, images = (
        payload["answer"],
        payload["chunks"],
        payload["images_used"],
    )
    if (
        not isinstance(answer, str)
        or not answer.strip()
        or not isinstance(chunks, list)
        or not chunks
        or not isinstance(images, list)
    ):
        raise ValueError("Incomplete cached response")
    for chunk in chunks:
        if (
            not isinstance(chunk, dict)
            or not isinstance(chunk.get("text"), str)
            or not chunk["text"]
            or not isinstance(chunk.get("payload"), dict)
            or not chunk.get("chunk_id")
            or not chunk.get("paper_id")
        ):
            raise ValueError("Invalid cached chunk")
    for image in images:
        if (
            not isinstance(image, dict)
            or not isinstance(image.get("path"), str)
            or not isinstance(image.get("n"), int)
            or image["n"] < 1
            or not any(
                c["chunk_id"] == image.get("chunk_id")
                and c["paper_id"] == image.get("paper_id")
                for c in chunks
            )
        ):
            raise ValueError("Invalid cached image reference")
        if images_root is None:
            raise ValueError("Images unavailable")
        root = images_root.resolve()
        path = (root / image["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Cached image unavailable")
    return {
        "answer": answer,
        "chunks": chunks,
        "images_used": images,
        "images_skipped": [],
        "candidates": [],
        "rerank_fallback": False,
        "image_fallback": False,
    }


class SemanticCache:
    def __init__(self, client: QdrantClient, options: CacheOptions):
        self.client, self.options = client, options

    async def _call(self, fn, **kwargs):
        return await asyncio.wait_for(
            asyncio.to_thread(fn, **kwargs), self.options.timeout_seconds
        )

    async def initialize(self):
        await self._call(
            ensure_cache_collection, client=self.client, name=self.options.collection
        )

    async def read(self, vector: list[float], version: str, top_k: int):
        result = await self._call(
            self.client.query_points,
            collection_name=self.options.collection,
            query=vector,
            using="dense",
            limit=1,
            with_payload=True,
            with_vectors=False,
            timeout=max(1, math.ceil(self.options.timeout_seconds)),
            score_threshold=self.options.threshold,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="cache_version", match=models.MatchValue(value=version)
                    ),
                    models.FieldCondition(
                        key="top_k", match=models.MatchValue(value=top_k)
                    ),
                    models.FieldCondition(
                        key="expires_at", range=models.Range(gt=int(time.time()))
                    ),
                ]
            ),
        )
        return result.points[0] if result.points else None

    async def write(self, vector: list[float], version: str, state: dict):
        now = int(time.time())
        payload = {key: state[key] for key in ("answer", "chunks", "images_used")}
        payload.update(
            cache_version=version,
            top_k=state["top_k"],
            created_at=now,
            expires_at=now + self.options.ttl_seconds,
        )
        point_id = str(uuid5(NAMESPACE_URL, version + ":" + state["query"].strip()))
        await self._call(
            self.client.upsert,
            collection_name=self.options.collection,
            points=[
                models.PointStruct(
                    id=point_id, vector={"dense": vector}, payload=payload
                )
            ],
            timeout=max(1, math.ceil(self.options.timeout_seconds)),
            wait=True,
        )

    async def cleanup(self):
        await self._call(
            self.client.delete,
            collection_name=self.options.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="expires_at", range=models.Range(lte=int(time.time()))
                        )
                    ]
                )
            ),
            timeout=max(1, math.ceil(self.options.timeout_seconds)),
            wait=True,
        )

    async def cleanup_loop(self):
        while True:
            try:
                await self.cleanup()
            except Exception:
                logger.exception("Semantic cache cleanup failed")
            await asyncio.sleep(3600)
