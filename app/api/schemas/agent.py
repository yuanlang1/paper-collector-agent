from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
)


AgentStatus = Literal[
    "running",
    "confirmation_required",
    "completed",
    "failed",
    "blocked",
]

ActionType = Literal[
    "tool",
    "subagent",
]

ActionResultStatus = Literal[
    "success",
    "accepted",
    "partial",
    "rejected",
    "error",
]


class PendingActionView(BaseModel):
    action_id: str
    action_type: ActionType
    kind: ActionType | None = None
    name: str
    display_name: str | None = None
    summary: str | None = None
    requires_confirmation: bool = False
    status: str | None = None


class ActionResultView(BaseModel):
    action_id: str
    action_type: ActionType
    name: str
    status: ActionResultStatus

    summary: str
    data: dict[str, Any] = Field(default_factory = dict)
    artifact_refs: list[str] = Field(default_factory = list)

    retryable: bool = False
    error_code: str | None = None
    error_message: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length = 1)
    conversation_id: str | None = None


class ChatResumeRequest(BaseModel):
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


class ChatResponse(BaseModel):
    conversation_id: str
    run_id: str

    status: AgentStatus
    reply: str = ""

    pending_action: PendingActionView | None

    last_action_result: ActionResultView | None

    artifact_refs: list[str] = Field(default_factory = list)

    interrupt: Any | None = None
    error: str | None = None
