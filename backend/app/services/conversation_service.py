"""DB persistence for conversations and messages.

Both functions only flush; the caller owns the transaction (commit/rollback),
so a whole turn is saved atomically.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import and_, func, or_, select, update
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


async def get_owned_conversation(
    db: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    user_id: int,
) -> Optional[Conversation]:
    """Ownership lives in the WHERE clause: a conversation that doesn't exist
    and one that belongs to someone else both come back as None."""
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def list_conversations(
    db: AsyncSession,
    *,
    user_id: int,
    limit: int = 20,
    before_updated_at: Optional[datetime] = None,
    before_id: Optional[uuid.UUID] = None,
) -> tuple[list[Conversation], bool]:
    """List one page, most recently active first, using a stable keyset cursor."""
    stmt = select(Conversation).where(Conversation.user_id == user_id)
    if before_updated_at is not None and before_id is not None:
        stmt = stmt.where(
            or_(
                Conversation.updated_at < before_updated_at,
                and_(
                    Conversation.updated_at == before_updated_at,
                    Conversation.id < before_id,
                ),
            )
        )

    result = await db.execute(
        stmt.order_by(Conversation.updated_at.desc(), Conversation.id.desc()).limit(limit + 1)
    )
    rows = list(result.scalars().all())
    return rows[:limit], len(rows) > limit


async def get_conversation_messages(
    db: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    limit: int,
    before_id: Optional[int] = None,
) -> tuple[list[Message], bool]:
    """Load the newest page before an optional message ID, oldest first."""
    stmt = select(Message).where(Message.conversation_id == conversation_id)
    if before_id is not None:
        stmt = stmt.where(Message.id < before_id)

    result = await db.execute(
        stmt.order_by(Message.id.desc()).limit(limit + 1)
    )
    rows = list(result.scalars().all())
    has_more = len(rows) > limit
    return list(reversed(rows[:limit])), has_more


def _paper_ids(sources: Optional[list[dict]]) -> list[str]:
    seen: dict[str, None] = {}
    for s in sources or []:
        if s.get("paper_id"):
            seen[s["paper_id"]] = None
    return list(seen)


async def get_history(
    db: AsyncSession, conversation_id: uuid.UUID, *, after_id: Optional[int] = None
) -> list[dict]:
    """User/assistant messages, oldest first. `after_id` skips messages already
    folded into the conversation summary."""
    stmt = select(Message.id, Message.role, Message.content, Message.sources).where(
        Message.conversation_id == conversation_id,
        Message.role.in_([MessageRole.user, MessageRole.assistant]),
    )
    if after_id is not None:
        stmt = stmt.where(Message.id > after_id)
    result = await db.execute(stmt.order_by(Message.created_at, Message.id))
    return [
        {"id": i, "role": r.value, "content": c, "paper_ids": _paper_ids(s)}
        for i, r, c, s in result.all()
    ]


async def save_summary(
    db: AsyncSession,
    conversation_id: uuid.UUID,
    *,
    summary: str,
    upto_message_id: int,
    expected_upto_id: Optional[int],
) -> bool:
    """Compare-and-set on the watermark so concurrent requests can't clobber each
    other. Returns False if someone else compacted first."""
    current = (
        Conversation.summary_upto_message_id.is_(None)
        if expected_upto_id is None
        else Conversation.summary_upto_message_id == expected_upto_id
    )
    result = await db.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id, current)
        .values(
            summary=summary,
            summary_upto_message_id=upto_message_id,
            # Set explicitly so onupdate=func.now() doesn't bump it; compaction
            # isn't user activity (touch_conversation does that per turn).
            updated_at=Conversation.updated_at,
        )
        .execution_options(synchronize_session=False)
    )
    return result.rowcount == 1  # type: ignore[attr-defined]


async def touch_conversation(db: AsyncSession, conversation_id: uuid.UUID) -> None:
    """Bump updated_at. Inserting messages doesn't UPDATE the conversation row,
    so onupdate=func.now() never fires on its own."""
    await db.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id)
        .values(updated_at=func.now())
        .execution_options(synchronize_session=False)
    )
