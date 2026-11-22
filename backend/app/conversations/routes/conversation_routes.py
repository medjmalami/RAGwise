from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.conversations.controllers import conversation_controller
from app.conversations.schemas import ConversationDetailResponse, ConversationListResponse
from app.db.db import get_session
from app.db.models import User, Message, Conversation
from pathlib import Path

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("/{conversation_id}/image")
async def get_image(
    conversation_id: UUID,
    request: Request,
    path: str = Query(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    authorized = await db.scalar(
        select(Message.id).join(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
            Message.sources.contains([{"images": [{"path": path}]}]),
        ).limit(1)
    )
    if authorized is None:
        raise HTTPException(404, "Image not found")
    root_value = getattr(request.app.state.llm, "images_root", None)
    if root_value is None:
        raise HTTPException(404, "Image not found")
    root = Path(root_value).resolve()
    target = (root / path).resolve()
    media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
    if not target.is_relative_to(root) or not target.is_file() or target.suffix.lower() not in media_types:
        raise HTTPException(404, "Image not found")
    return FileResponse(target, media_type=media_types[target.suffix.lower()], headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})


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
