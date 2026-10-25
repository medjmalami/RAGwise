from uuid import UUID
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.conversations.schemas import (
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationMessageResponse,
    ConversationResponse,
)
from app.db.models import User
from app.services.conversation_service import (
    get_conversation_messages,
    get_owned_conversation,
    list_conversations,
)


async def get_conversations(
    db: AsyncSession,
    *,
    user: User,
    limit: int = 20,
    before_updated_at: datetime | None = None,
    before_id: UUID | None = None,
) -> ConversationListResponse:
    conversations, has_more = await list_conversations(
        db,
        user_id=user.id,
        limit=limit,
        before_updated_at=before_updated_at,
        before_id=before_id,
    )
    return ConversationListResponse(
        conversations=[ConversationResponse.model_validate(item) for item in conversations],
        has_more=has_more,
        next_before_updated_at=conversations[-1].updated_at if has_more and conversations else None,
        next_before_id=conversations[-1].id if has_more and conversations else None,
    )


async def get_conversation(
    db: AsyncSession,
    *,
    conversation_id: UUID,
    user: User,
    limit: int = 10,
    before_id: int | None = None,
) -> ConversationDetailResponse:
    conversation = await get_owned_conversation(
        db, conversation_id=conversation_id, user_id=user.id
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages, has_more = await get_conversation_messages(
        db,
        conversation_id=conversation_id,
        limit=limit,
        before_id=before_id,
    )
    return ConversationDetailResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[ConversationMessageResponse.model_validate(item) for item in messages],
        has_more=has_more,
        next_before_id=messages[0].id if has_more and messages else None,
    )
