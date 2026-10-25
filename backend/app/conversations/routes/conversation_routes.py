from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.conversations.controllers import conversation_controller
from app.conversations.schemas import ConversationDetailResponse, ConversationResponse
from app.db.db import get_session
from app.db.models import User

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationResponse])
async def get_conversations(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    return await conversation_controller.get_conversations(db, user=user)


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    return await conversation_controller.get_conversation(
        db, conversation_id=conversation_id, user=user
    )
