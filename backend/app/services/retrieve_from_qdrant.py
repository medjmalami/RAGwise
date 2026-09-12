"""
Retrieval helpers for the RAGwise pipeline.

  * embed_query()    — embed a user query with BAAI/bge-m3 (dense + sparse).
  * retrieve_top_k() — hybrid (dense + sparse) retrieval over the Qdrant
                       collection populated by embed_to_qdrant.py.

Conventions mirrored from embed_to_qdrant.py:
    * Qdrant dense vector name : "dense"  (1024-d, cosine)
    * Qdrant sparse vector name: "sparse" (BGE-M3 lexical weights)
    * payload fields           : paper_id, has_picture, chunk_id, text, ...

BGE-M3 does NOT require an instruction prefix for retrieval, so queries are
encoded with the same encode() settings used for documents.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, cast

from qdrant_client import QdrantClient, models

# FlagEmbedding/torch/numpy are heavy imports, so they're loaded lazily at
# runtime inside QueryEmbedder.load(). This TYPE_CHECKING import gives the
# type checker real symbols for annotations/casts without paying that cost
# at import time.
if TYPE_CHECKING:
    import numpy as np
    from FlagEmbedding import BGEM3FlagModel

DENSE_DIM = 1024
DEFAULT_COLLECTION = "ragwise_arxiv"
DEFAULT_MAX_LENGTH = 640  # Match the value used at chunk-embedding time.
DEFAULT_PREFETCH_LIMIT = 20  # Candidates each prefetch pulls before fusion.


# ---------------------------------------------------------------------------
# Model wrapper — load BGEM3FlagModel once, reuse across queries.
# ---------------------------------------------------------------------------
@dataclass
class QueryEmbedder:
    """Wraps a BGEM3FlagModel so we pay the load cost once.

    In a long-running service: build this at startup, share it across
    requests. In a notebook: just call `.load()` once at the top.
    """

    model_name: str = "BAAI/bge-m3"
    device: Optional[str] = None  # None → auto (cuda if available else cpu)
    use_fp16: bool = True  # Ignored on CPU.
    max_length: int = DEFAULT_MAX_LENGTH
    _model: Optional["BGEM3FlagModel"] = field(default=None, init=False, repr=False)

    def load(self) -> "QueryEmbedder":
        import torch
        from FlagEmbedding import BGEM3FlagModel

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        use_fp16 = self.use_fp16 and device == "cuda"
        print(f"Loading {self.model_name} on {device} (fp16={use_fp16})...")
        self._model = BGEM3FlagModel(self.model_name, use_fp16=use_fp16, device=device)
        return self

    @property
    def model(self) -> "BGEM3FlagModel":
        if self._model is None:
            self.load()
        # Re-check explicitly (rather than trusting the branch above) so the
        # type checker re-narrows self._model to non-None here, instead of
        # inferring across the self.load() call — that cross-call inference
        # is what was collapsing this property's return type to None.
        if self._model is None:
            raise RuntimeError(
                f"Failed to load {self.model_name}: self._model is still None "
                "after calling load()."
            )
        return self._model


# ---------------------------------------------------------------------------
# Function 1: embed a query.
# ---------------------------------------------------------------------------
def embed_query(
    query: str,
    embedder: QueryEmbedder,
    return_sparse: bool = True,
) -> dict:
    """Embed a single query string with BGE-M3.

    Args:
        query:         Raw user query. No prefix needed for BGE-M3.
        embedder:      A loaded QueryEmbedder.
        return_sparse: Set False if the target collection is dense-only.

    Returns:
        {
            "dense":  list[float] of length 1024,
            "sparse": dict[str, float] {token_id_str: weight}  (if return_sparse),
        }

    Notes:
        * Query is wrapped in a 1-element list because BGEM3FlagModel.encode
          expects a sequence. Batch_size=1 is fine — for batched query
          embedding (e.g. expanding one query into multiple rewrites), pass
          them all at once and bump batch_size.
        * `max_length` is taken from the embedder and should match the value
          used at index time. Mismatching it doesn't break anything but does
          slightly perturb cosine scores (longer queries get truncated
          differently).
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    out = embedder.model.encode(
        [query],
        batch_size=1,
        max_length=embedder.max_length,
        return_dense=True,
        return_sparse=return_sparse,
        return_colbert_vecs=False,
    )

    # BGEM3FlagModel.encode()'s return annotation is a plain Dict[str, ...]
    # with one Union value type shared across every key, rather than a
    # TypedDict with a distinct type per key. So the checker treats
    # out["dense_vecs"][0] as possibly being the same type as
    # out["lexical_weights"][0] (a Dict[str, float]) — hence "tolist" not
    # existing on that branch. Which key holds which shape is fixed by the
    # return_dense/return_sparse flags we passed above, so cast each
    # extraction to what's actually there.
    dense = cast("np.ndarray", out["dense_vecs"])[0]
    dense = dense.tolist() if hasattr(dense, "tolist") else list(dense)

    result: dict = {"dense": dense}
    if return_sparse:
        # BGE-M3 returns {token_id_str: weight}; keep that shape here and
        # convert to Qdrant's SparseVector only at retrieval time.
        lexical_weights = cast("list[dict[str, float]]", out["lexical_weights"])
        result["sparse"] = lexical_weights[0] or {}
    return result


# ---------------------------------------------------------------------------
# Function 2: retrieve top-k chunks via hybrid (dense + sparse) search.
# ---------------------------------------------------------------------------
def _to_sparse_vector(lexical_weights: dict) -> models.SparseVector:
    """Mirror of embed_to_qdrant.to_sparse_vector: indices sorted ascending.

    Some Qdrant versions are picky about ascending indices; sorting is cheap
    and avoids a class of bugs that's hard to spot.
    """
    if not lexical_weights:
        return models.SparseVector(indices=[], values=[])
    pairs = sorted(
        ((int(tok), float(w)) for tok, w in lexical_weights.items()),
        key=lambda p: p[0],
    )
    return models.SparseVector(
        indices=[p[0] for p in pairs], values=[p[1] for p in pairs]
    )


def retrieve_top_k(
    query: str,
    client: QdrantClient,
    embedder: QueryEmbedder,
    collection: str = DEFAULT_COLLECTION,
    k: int = 5,
    use_sparse: bool = True,
    query_filter: Optional[models.Filter] = None,
    score_threshold: Optional[float] = None,
    prefetch_limit: int = DEFAULT_PREFETCH_LIMIT,
) -> list[dict]:
    """Hybrid retrieval of the top-k chunks for `query`.

    Strategy:
        1. Dense prefetch   — cosine search on the "dense" named vector.
        2. Sparse prefetch  — BM25-style search on the "sparse" named vector.
        3. RRF fusion       — Qdrant fuses the two ranked lists by reciprocal
           rank. RRF is the right default because dense cosine (~[-1, 1])
           and sparse BM25 (~[0, ∞)) live on incomparable scales; rank-based
           fusion sidesteps the weight-tuning problem entirely.

    Args:
        query:           User query string.
        client:          An open QdrantClient (gRPC or REST — both fine).
        embedder:        A loaded QueryEmbedder.
        collection:      Qdrant collection name.
        k:               Final number of chunks to return.
        use_sparse:      False if the collection was built with --no-sparse.
        query_filter:    Optional Qdrant Filter, e.g. restrict to one paper:
                             models.Filter(must=[models.FieldCondition(
                                 key="paper_id",
                                 match=models.MatchValue(value="2401.12345"),
                             )])
                         or picture-only chunks:
                             models.Filter(must=[models.FieldCondition(
                                 key="has_picture",
                                 match=models.MatchValue(value=True),
                             )])
        score_threshold: Drop results whose fused score is below this.
        prefetch_limit:  Candidates each prefetch pulls before fusion.
                         Bump to 50–100 if you filter heavily — the filter
                         applies inside the prefetch, so a tight filter on a
                         small prefetch can starve the fusion.

    Returns:
        List of dicts sorted by descending fused score, each shaped:
            {
                "score":    float,            # fused RRF score
                "chunk_id": str | None,
                "paper_id": str | None,
                "text":     str | None,
                "payload":  {...full chunk payload...},
            }
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    embedded = embed_query(query, embedder, return_sparse=use_sparse)
    dense_vec = embedded["dense"]

    prefetch: list[models.Prefetch] = [
        models.Prefetch(
            query=dense_vec,
            using="dense",
            limit=prefetch_limit,
            filter=query_filter,
        )
    ]
    if use_sparse:
        sparse_vec = _to_sparse_vector(embedded["sparse"])
        prefetch.append(
            models.Prefetch(
                query=sparse_vec,
                using="sparse",
                limit=prefetch_limit,
                filter=query_filter,
            )
        )

    # When `prefetch` has >1 entry and `query` is None, Qdrant applies RRF
    # fusion automatically. `using` is required at the top level by the API
    # but is ignored during pure fusion — pass "dense" as a harmless default.
    response = client.query_points(
        collection_name=collection,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True,
        with_vectors=False,
    )

    results = []
    for point in response.points:
        payload = point.payload or {}
        if score_threshold is not None and point.score < score_threshold:
            continue
        results.append(
            {
                "score": float(point.score),
                "chunk_id": payload.get("chunk_id"),
                "paper_id": payload.get("paper_id"),
                "text": payload.get("text"),
                "payload": payload,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Example usage.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Build clients once. In a service: do this in app startup, not per request.
    embedder = QueryEmbedder().load()

    client = QdrantClient(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY"),
        prefer_grpc=True,
        grpc_port=6334,
    )

    query = "What is the asymptotic regret of Thompson sampling for linear bandits?"

    # 1. Just the embedding (e.g. to log, cache, or hand to a reranker).
    vec = embed_query(query, embedder)
    print(f"dense dim: {len(vec['dense'])}, sparse nnz: {len(vec['sparse'])}")

    # 2. Top-k chunks, hybrid retrieval.
    hits = retrieve_top_k(
        query=query,
        client=client,
        embedder=embedder,
        collection="ragwise_arxiv",
        k=5,
    )
    for h in hits:
        print(f"[{h['score']:.4f}] {h['paper_id']} :: {h['text'][:160]}...")

    # 3. Same query, but scoped to one paper via payload filter.
    scoped = retrieve_top_k(
        query=query,
        client=client,
        embedder=embedder,
        k=3,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="paper_id",
                    match=models.MatchValue(value="2401.12345"),
                )
            ]
        ),
    )
