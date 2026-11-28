"""Run from backend: PYTHONPATH=. .venv/bin/python scripts/test_source_images.py."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, create_autospec
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession

from app.conversations.routes.conversation_routes import get_image
from app.db.models import User
from app.services.conversation_service import slim_sources


class ImageAccessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.app = FastAPI()
        self.request = Request({"type": "http", "app": self.app})
        self.user = User(id=42)
        session = create_autospec(AsyncSession, instance=True)
        self.db: AsyncSession = session
        self.scalar_mock: AsyncMock = session.scalar

    async def test_saved_image_and_query_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "figure.png"
            image.touch()
            self.app.state.llm = SimpleNamespace(images_root=directory)
            self.scalar_mock.return_value = 1
            response = await get_image(
                uuid4(), self.request, "figure.png", self.user, self.db
            )
            self.assertEqual(response.path, image)
            self.assertEqual(response.media_type, "image/png")
            self.scalar_mock.assert_awaited_once()
            call = self.scalar_mock.await_args
            assert call is not None
            statement = call.args[0].compile(dialect=postgresql.dialect())
            self.assertIn("conversations.user_id =", str(statement))
            self.assertIn("conversations.id =", str(statement))
            self.assertIn("@>", str(statement))

    async def test_unreferenced_missing_unsupported_and_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "figure.svg").touch()
            self.app.state.llm = SimpleNamespace(images_root=directory)
            for path, authorized in [
                ("figure.png", None),
                ("missing.png", 1),
                ("figure.svg", 1),
                ("../outside.png", 1),
            ]:
                with self.subTest(path=path):
                    self.scalar_mock.return_value = authorized
                    with self.assertRaises(HTTPException) as error:
                        await get_image(uuid4(), self.request, path, self.user, self.db)
                    self.assertEqual(error.exception.status_code, 404)

    def test_image_references_persist_and_legacy_sources_default_empty(self):
        images = [{"path": "paper/figure.png", "n": 1}]
        self.assertEqual(slim_sources([{"images": images}])[0]["images"], images)
        self.assertEqual(slim_sources([{}])[0]["images"], [])


if __name__ == "__main__":
    unittest.main()
