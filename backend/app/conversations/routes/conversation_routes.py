from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.conversations.controllers import conversation_controller
from app.conversations.schemas import ConversationDetailResponse, ConversationListResponse
from app.db.db import get_session
from app.db.models import User

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=ConversationListResponse)
async def get_conversations(
    limit: int = Query(default=20, ge=1, le=100),
    before_updated_at: datetime | None = Query(default=None),
    before_id: UUID | None = Query(default=None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    return await conversation_controller.get_conversations(
        db,
        user=user,
        limit=limit,
        before_updated_at=before_updated_at,
        before_id=before_id,
    )


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: UUID,
    limit: int = Query(default=10, ge=1, le=100),
    before_id: int | None = Query(default=None, gt=0),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    return await conversation_controller.get_conversation(
        db,
        conversation_id=conversation_id,
        user=user,
        limit=limit,
        before_id=before_id,
    )
