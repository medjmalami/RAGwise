"""Run from backend: PYTHONPATH=. .venv/bin/python scripts/test_source_images.py."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from app.conversations.routes.conversation_routes import get_image
from app.services.conversation_service import slim_sources


class ImageAccessTests(unittest.IsolatedAsyncioTestCase):
    async def test_saved_image_and_query_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "figure.png"
            image.touch()
            db = SimpleNamespace(scalar=AsyncMock(return_value=1))
            request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(llm=SimpleNamespace(images_root=directory))))
            response = await get_image(uuid4(), request, "figure.png", SimpleNamespace(id=42), db)
            self.assertEqual(response.path, image)
            self.assertEqual(response.media_type, "image/png")
            statement = db.scalar.call_args.args[0].compile(dialect=postgresql.dialect())
            self.assertIn("conversations.user_id =", str(statement))
            self.assertIn("conversations.id =", str(statement))
            self.assertIn("@>", str(statement))

    async def test_unreferenced_missing_unsupported_and_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "figure.svg").touch()
            request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(llm=SimpleNamespace(images_root=directory))))
            for path, authorized in [("figure.png", None), ("missing.png", 1), ("figure.svg", 1), ("../outside.png", 1)]:
                with self.subTest(path=path):
                    db = SimpleNamespace(scalar=AsyncMock(return_value=authorized))
                    with self.assertRaises(HTTPException) as error:
                        await get_image(uuid4(), request, path, SimpleNamespace(id=42), db)
                    self.assertEqual(error.exception.status_code, 404)

    def test_image_references_persist_and_legacy_sources_default_empty(self):
        images = [{"path": "paper/figure.png", "n": 1}]
        self.assertEqual(slim_sources([{"images": images}])[0]["images"], images)
        self.assertEqual(slim_sources([{}])[0]["images"], [])


if __name__ == "__main__":
    unittest.main()
