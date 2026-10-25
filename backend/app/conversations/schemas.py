from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.db.models import MessageRole


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    conversations: list[ConversationResponse]
    has_more: bool
    next_before_updated_at: datetime | None
    next_before_id: UUID | None


class ConversationMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: MessageRole
    content: str
    sources: list[dict] | None = None
    created_at: datetime


class ConversationDetailResponse(ConversationResponse):
    messages: list[ConversationMessageResponse]
    has_more: bool
    next_before_id: int | None
