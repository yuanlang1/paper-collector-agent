from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


def _validate_user_id(value: str) -> str:
    if int(value) > 9_223_372_036_854_775_807:
        raise ValueError("user_id must be a non-negative BIGINT")
    return str(int(value))


class ChatRequest(BaseModel):
    user_id: str = Field(default="0", min_length=1, max_length=19, pattern=r"^\d+$")
    conversation_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length = 1)
    llm_profile_id: int | None = Field(default=None, gt=0)

    _user_id = field_validator("user_id")(_validate_user_id)


class ChatResumeRequest(BaseModel):
    user_id: str = Field(default="0", min_length=1, max_length=19, pattern=r"^\d+$")
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

    _user_id = field_validator("user_id")(_validate_user_id)


class RunResumeRequest(BaseModel):
    user_id: str = Field(default="0", min_length=1, max_length=19, pattern=r"^\d+$")
    conversation_id: str = Field(..., min_length=1)
    action_id: str = Field(..., min_length=1)
    decision: Literal[
        "approved",
        "rejected",
    ]
    query_understanding: dict[str, Any] | None = None
    search_tag: dict[str, Any] | None = None
    comment: str | None = None

    _user_id = field_validator("user_id")(_validate_user_id)


class RunStartResponse(BaseModel):
    conversation_id: str
    run_id: str
    status: Literal["started", "resumed"]
