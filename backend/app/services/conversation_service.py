"""DB persistence for conversations and messages.

Both functions only flush; the caller owns the transaction (commit/rollback),
so a whole turn is saved atomically.
"""

from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, Message, MessageRole

TITLE_MAX_LEN = 80


def build_title(query: str) -> str:
    """Title = the first ~80 chars of the query, whitespace collapsed."""
    title = " ".join(query.split())
    if len(title) > TITLE_MAX_LEN:
        title = title[: TITLE_MAX_LEN - 1].rstrip() + "…"
    return title


def slim_sources(chunks: list[dict]) -> list[dict]:
    """What we keep in Message.sources: references, not chunk text/payload.

    `score` is the Cohere relevance score. `retrieval_score` is the original
    hybrid (RRF) score; it is None when the rerank step failed and the graph
    fell back to hybrid order (in that case `score` is the RRF score).
    """
    return [
        {
            "paper_id": c.get("paper_id"),
            "chunk_id": c.get("chunk_id"),
            "score": c.get("score"),
            "retrieval_score": c.get("retrieval_score"),
            "hybrid_rank": c.get("hybrid_rank"),
        }
        for c in chunks
    ]


async def create_conversation(
    db: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    user_id: int,
    title: Optional[str],
) -> Conversation:
    conversation = Conversation(id=conversation_id, user_id=user_id, title=title)
    db.add(conversation)
    # Flush so the row exists before messages reference it.
    await db.flush()
    return conversation


async def add_message(
    db: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    role: MessageRole,
    content: str,
    sources: Optional[list[dict]] = None,
    trace_id: Optional[str] = None,
) -> Message:
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources=sources,
        trace_id=trace_id,
    )
    db.add(message)
    # Flush per message: created_at is now() = transaction start in Postgres,
    # so messages in one transaction tie on it. Ordering relies on the id
    # tiebreaker, which needs strictly increasing ids in insertion order.
    await db.flush()
    return message
