"""ChatOllama that turns image *paths* into base64 at the last possible moment.

Why this exists
---------------
LangChain fires its callbacks (and therefore Langfuse) with the messages
exactly as the caller built them, *before* the model's _generate/_stream runs.
So if the graph puts light `image_path` blocks in the message and this class
converts them to base64 inside _generate/_stream, the trace only ever sees the
paths -- never the bytes.

Block format the graph sends:
    {"type": "image_path", "path": "<relative to images_root>", ...extra keys}

Extra keys (chunk_id, paper_id, n) are ignored here; they only make the trace
more useful.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, TypeGuard

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_ollama import ChatOllama

IMAGE_PATH_BLOCK = "image_path"

# backend/app/services/vision_llm.py -> parents[2] == backend/
DEFAULT_IMAGES_ROOT = (
    Path(__file__).resolve().parents[2] / "arxiv_rag_papers" / "parsed"
)


def image_path_block(path: str, **meta: Any) -> dict:
    return {"type": IMAGE_PATH_BLOCK, "path": path, **meta}


def _is_path_block(block: Any) -> TypeGuard[dict[str, Any]]:
    return isinstance(block, dict) and block.get("type") == IMAGE_PATH_BLOCK


def _to_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


class PathImageChatOllama(ChatOllama):
    """ChatOllama that accepts `image_path` content blocks."""

    images_root: Path = DEFAULT_IMAGES_ROOT

    def _resolve_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        root = self.images_root.resolve()
        resolved: list[BaseMessage] = []
        for m in messages:
            content = m.content
            if isinstance(content, list) and any(_is_path_block(b) for b in content):
                blocks: list[Any] = []
                for b in content:
                    if not _is_path_block(b):
                        blocks.append(b)
                        continue
                    p = (root / b["path"]).resolve()
                    if not p.is_relative_to(root):
                        raise ValueError(f"image path escapes images_root: {b['path']}")
                    blocks.append(
                        {"type": "image_url", "image_url": {"url": _to_data_uri(p)}}
                    )
                m = m.model_copy(update={"content": blocks})
            resolved.append(m)
        return resolved

    # The four overrides below only swap path blocks for base64 and delegate.
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return super()._generate(
            self._resolve_messages(messages),
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        resolved = await asyncio.to_thread(self._resolve_messages, messages)
        return await super()._agenerate(
            resolved, stop=stop, run_manager=run_manager, **kwargs
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        yield from super()._stream(
            self._resolve_messages(messages),
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        resolved = await asyncio.to_thread(self._resolve_messages, messages)
        async for chunk in super()._astream(
            resolved, stop=stop, run_manager=run_manager, **kwargs
        ):
            yield chunk
