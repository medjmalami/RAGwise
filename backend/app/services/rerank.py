from __future__ import annotations

import cohere

DEFAULT_RERANK_MODEL = "rerank-v3.5"


class CohereReranker:
    """Thin async wrapper around Cohere Rerank (v2 API).

    Build once at startup and share across requests.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_RERANK_MODEL,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self._client = cohere.AsyncClientV2(api_key=api_key, timeout=timeout)

    async def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_n: int,
        min_score: float | None = None,
    ) -> list[dict]:
        # Cohere rejects empty/None documents, so drop chunks without text.
        docs = [c for c in chunks if c.get("text")]
        if not docs:
            return []

        response = await self._client.rerank(
            model=self.model,
            query=query,
            documents=[c["text"] for c in docs],
            top_n=min(top_n, len(docs)),
        )

        reranked: list[dict] = []
        for r in response.results:  # already sorted by relevance, descending
            if min_score is not None and r.relevance_score < min_score:
                continue
            chunk = docs[r.index]
            reranked.append(
                {
                    **chunk,
                    "retrieval_score": chunk["score"],  # original RRF score
                    "score": float(r.relevance_score),  # Cohere relevance
                }
            )
        return reranked
