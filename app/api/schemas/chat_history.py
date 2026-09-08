from typing import Any, Literal

from pydantic import BaseModel, Field


class ConversationView(BaseModel):
    conversation_id: str
    title: str
    last_message_preview: str
    message_count: int
    last_message_at: str


class ConversationListResponse(BaseModel):
    items: list[ConversationView] = Field(default_factory=list)


class ChatMessageView(BaseModel):
    id: int
    conversation_id: str
    run_id: str
    role: Literal["user", "assistant"]
    content: str
    status: str
    source: str
    meta: dict[str, Any] | None = None
    created_at: str


class ChatMessageListResponse(BaseModel):
    conversation_id: str
    items: list[ChatMessageView] = Field(default_factory=list)
    next_before_id: int | None = None


class DeleteConversationResponse(BaseModel):
    conversation_id: str
    deleted: bool = True
