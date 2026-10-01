from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
)


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    conversation_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length = 1)
    llm_profile_id: int | None = Field(default=None, gt=0)


class ChatResumeRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    conversation_id: str = Field(..., min_length = 1)
    run_id: str = Field(..., min_length=1)
    action_id: str = Field(..., min_length=1)
    decision: Literal[
        "approved",
        "rejected",
    ]
    query_understanding: dict[str, Any] | None = None
    search_tag: dict[str, Any] | None = None
    comment: str | None = None
