"""Run from backend: PYTHONPATH=. .venv/bin/python scripts/test_semantic_cache.py."""

import asyncio
import copy
import os
import math
import tempfile
import time
import unittest
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock, create_autospec, patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import QueryResponse
from app.graph.rag_graph import build_rag_graph
from app.services.rerank import CohereReranker
from app.services.retrieve_from_qdrant import QueryEmbedder, retrieve_top_k
from app.services.semantic_cache import (
    CacheOptions,
    SemanticCache,
    validate_payload,
    write_skip_reason,
)

VECTOR = [1.0] + [0.0] * 1023
EMBEDDED = {"dense": VECTOR, "sparse": {"1": 0.5}}
CHUNKS = [
    {
        "chunk_id": "c1",
        "paper_id": "p1",
        "text": "Evidence",
        "score": 0.9,
        "payload": {},
    }
]
INPUT = {
    "query": "What is the evidence?",
    "top_k": 5,
    "paper_id": None,
    "has_picture": None,
    "history": [],
}


class CacheTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        telemetry = patch("app.graph.rag_graph.get_client")
        telemetry.start()
        self.addCleanup(telemetry.stop)
        propagation = patch(
            "app.graph.rag_graph.propagate_attributes",
            side_effect=lambda **kw: nullcontext(),
        )
        propagation.start()
        self.addCleanup(propagation.stop)
        self.live = os.environ.get("RAGWISE_TEST_QDRANT_URL")
        self.client = (
            QdrantClient(url=self.live, timeout=2)
            if self.live
            else QdrantClient(":memory:")
        )
        self.cache = SemanticCache(
            self.client, CacheOptions(collection="ragwise_cache_test_" + uuid4().hex)
        )
        await self.cache.initialize()
        reranker = create_autospec(CohereReranker, instance=True)
        reranker.model = "test"
        self.rerank_mock: AsyncMock = reranker.rerank
        self.rerank_mock.side_effect = lambda **kwargs: kwargs["chunks"]
        self.reranker: CohereReranker = reranker
        self.llm = FakeListChatModel(responses=["Supported answer [1]."])

    async def asyncTearDown(self):
        if self.live:
            self.client.delete_collection(self.cache.options.collection)
        self.client.close()

    def graph(self, **kwargs):
        return build_rag_graph(
            QueryEmbedder(),
            self.client,
            self.llm,
            self.reranker,
            use_images=False,
            cache=self.cache,
            **kwargs,
        )

    async def test_miss_hit_followup(self):
        graph = self.graph(use_cache=True)
        with (
            patch("app.graph.rag_graph.embed_query", return_value=EMBEDDED) as embed,
            patch(
                "app.graph.rag_graph.retrieve_top_k",
                side_effect=lambda **kw: copy.deepcopy(CHUNKS),
            ) as retrieve,
        ):
            first = await graph.ainvoke(INPUT)
            self.assertEqual(
                (first["cache_status"], first["cache_write_status"]),
                ("miss", "written"),
            )
            self.assertEqual(embed.call_count, 1)
            self.assertEqual(retrieve.call_args.kwargs["embedded"], EMBEDDED)
            second = await graph.ainvoke(INPUT)
            self.assertEqual(second["cache_status"], "hit")
            self.assertEqual(second["answer"], first["answer"])
            self.assertEqual(second["chunks"], first["chunks"])
            self.assertEqual(retrieve.call_count, 1)
            self.rerank_mock.assert_awaited_once()
            for history, summary in [
                ([{"role": "user", "content": "Earlier"}], None),
                ([], "Earlier conversation"),
            ]:
                result = await graph.ainvoke(
                    {**INPUT, "history": history, "history_summary": summary}
                )
                self.assertEqual(result["cache_status"], "skipped")
                self.assertIsNone(retrieve.call_args.kwargs["embedded"])
            self.assertEqual(embed.call_count, 2)

    async def test_configuration_changes_do_not_hit(self):
        with (
            patch("app.graph.rag_graph.embed_query", return_value=EMBEDDED),
            patch(
                "app.graph.rag_graph.retrieve_top_k",
                side_effect=lambda **kw: copy.deepcopy(CHUNKS),
            ),
        ):
            await self.graph(use_cache=True).ainvoke(INPUT)
            self.reranker.model = "different-reranker"
            self.assertEqual(
                (await self.graph(use_cache=True).ainvoke(INPUT))["cache_status"],
                "miss",
            )
            self.assertEqual(
                (await self.graph(use_cache=True).ainvoke({**INPUT, "paper_id": "p1"}))[
                    "cache_status"
                ],
                "miss",
            )

    async def test_disabled_and_failures(self):
        with (
            patch(
                "app.graph.rag_graph.retrieve_top_k", return_value=copy.deepcopy(CHUNKS)
            ),
            patch("app.graph.rag_graph.embed_query", return_value=EMBEDDED) as embed,
        ):
            result = await self.graph(use_cache=False).ainvoke(INPUT)
            self.assertEqual(result["cache_status"], "skipped")
            embed.assert_not_called()
            with (
                patch.object(self.cache, "read", side_effect=RuntimeError("offline")),
                patch.object(self.cache, "write", side_effect=RuntimeError("offline")),
            ):
                result = await self.graph(use_cache=True).ainvoke(INPUT)
            self.assertEqual(
                (result["cache_status"], result["cache_write_status"]),
                ("error", "error"),
            )
            self.assertEqual(result["answer"], "Supported answer [1].")

    async def test_expiry_scope_threshold_cleanup(self):
        state = {**INPUT, "answer": "Answer", "chunks": CHUNKS, "images_used": []}
        with patch("app.services.semantic_cache.time.time", return_value=100):
            await self.cache.write(VECTOR, "v1", state)
            self.assertIsNotNone(await self.cache.read(VECTOR, "v1", 5))
            for similarity, expected in [(0.919, False), (0.921, True)]:
                vector = [similarity, math.sqrt(1 - similarity**2)] + [0.0] * 1022
                self.assertEqual(
                    await self.cache.read(vector, "v1", 5) is not None, expected
                )
            self.assertIsNone(await self.cache.read(VECTOR, "v2", 5))
            self.assertIsNone(await self.cache.read(VECTOR, "v1", 3))
            self.assertIsNone(await self.cache.read([0.0, 1.0] + [0.0] * 1022, "v1", 5))
        with patch(
            "app.services.semantic_cache.time.time",
            return_value=100 + self.cache.options.ttl_seconds,
        ):
            self.assertIsNone(await self.cache.read(VECTOR, "v1", 5))
            await self.cache.cleanup()
        self.assertEqual(self.client.count(self.cache.options.collection).count, 0)

    async def test_invalid_payload_and_rerank_fallback(self):
        hit = SimpleNamespace(score=0.99, payload={"answer": "broken"})
        with (
            patch.object(self.cache, "read", return_value=hit),
            patch.object(self.cache, "write", new_callable=AsyncMock) as write,
            patch("app.graph.rag_graph.embed_query", return_value=EMBEDDED),
            patch(
                "app.graph.rag_graph.retrieve_top_k", return_value=copy.deepcopy(CHUNKS)
            ),
            patch.object(self.reranker, "rerank", side_effect=RuntimeError("offline")),
        ):
            result = await self.graph(use_cache=True).ainvoke(INPUT)
            self.assertEqual(result["cache_status"], "miss")
            self.assertEqual(result["cache_skip_reason"], "fallback")
            write.assert_not_called()

    async def test_no_docs_and_abstention(self):
        for chunks, answer in [
            ([], "unused"),
            (CHUNKS, "I don't know."),
            (CHUNKS, "I cannot determine this from the context."),
        ]:
            self.llm = FakeListChatModel(responses=[answer])
            with (
                patch("app.graph.rag_graph.embed_query", return_value=EMBEDDED),
                patch(
                    "app.graph.rag_graph.retrieve_top_k",
                    return_value=copy.deepcopy(chunks),
                ),
                patch.object(self.cache, "write", new_callable=AsyncMock) as write,
            ):
                result = await self.graph(use_cache=True).ainvoke(INPUT)
                self.assertEqual(result["cache_write_status"], "skipped")
                write.assert_not_called()

    async def test_timeout_and_cancellation(self):
        cache = SemanticCache(self.client, CacheOptions(timeout_seconds=0.01))
        with self.assertRaises(TimeoutError):
            await cache._call(lambda: time.sleep(0.1))
        task = asyncio.create_task(cache.cleanup_loop())
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    def test_image_validation_and_exclusions(self):
        payload = {
            "answer": "Answer [1]",
            "chunks": CHUNKS,
            "images_used": [
                {"chunk_id": "c1", "paper_id": "p1", "n": 1, "path": "figure.png"}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                validate_payload(payload, root)
            (root / "figure.png").touch()
            self.assertEqual(
                validate_payload(payload, root)["images_used"], payload["images_used"]
            )
            payload["images_used"][0]["path"] = "../outside.png"
            with self.assertRaises(ValueError):
                validate_payload(payload, root)
        for extra in [
            {"image_fallback": True},
            {"rerank_fallback": True},
            {"answer": ""},
            {"answer": "I don’t know."},
            {"chunks": []},
        ]:
            self.assertIsNotNone(
                write_skip_reason({"answer": "Answer", "chunks": CHUNKS, **extra})
            )

    async def test_cached_sse_and_persistence(self):
        from app.controllers import controller

        images = [{"chunk_id": "c1", "paper_id": "p1", "path": "figure.png", "n": 1}]

        async def stream(*args, **kwargs):
            yield (
                "updates",
                {
                    "read_cache": {
                        "cache_status": "hit",
                        "answer": "Cached [1]",
                        "chunks": copy.deepcopy(CHUNKS),
                        "images_used": images,
                    }
                },
            )

        graph = self.graph(use_cache=True)
        ctx = controller.TurnContext(
            query=INPUT["query"],
            is_new=True,
            history=[],
            existing_title=None,
            summary=None,
            summary_upto_id=None,
            user_id=1,
            conversation_id=uuid4(),
        )
        queue = asyncio.Queue()
        with (
            patch.object(graph, "astream", new=stream),
            patch.object(
                controller, "_make_title", new_callable=AsyncMock, return_value="Title"
            ),
            patch.object(
                controller, "_persist_turn", new_callable=AsyncMock
            ) as persist,
            patch.object(
                controller,
                "CallbackHandler",
                return_value=SimpleNamespace(last_trace_id="trace"),
            ),
            patch.object(
                controller,
                "propagate_attributes",
                side_effect=lambda **kw: nullcontext(),
            ),
        ):
            await controller._produce(ctx, graph, self.llm, self.llm, queue)
        events = []
        while not queue.empty():
            item = queue.get_nowait()
            if item is not controller._END:
                events.append(item)
        self.assertEqual([event for event, _ in events], ["sources", "token", "done"])
        self.assertEqual(events[1][1], {"text": "Cached [1]"})
        self.assertEqual(
            events[2][1]["sources"][0]["images"], [{"path": "figure.png", "n": 1}]
        )
        self.assertEqual(persist.call_args.kwargs["answer"], "Cached [1]")
        self.assertEqual(
            persist.call_args.kwargs["chunks"][0]["images"],
            [{"path": "figure.png", "n": 1}],
        )

    def test_retrieval_embedding_reuse(self):
        with (
            patch.object(
                self.client, "query_points", return_value=QueryResponse(points=[])
            ),
            patch("app.services.retrieve_from_qdrant.embed_query") as embed,
        ):
            retrieve_top_k("query", self.client, QueryEmbedder(), embedded=EMBEDDED)
            embed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
