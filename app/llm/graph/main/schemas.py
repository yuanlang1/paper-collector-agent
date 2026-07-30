from __future__ import annotations

from typing import Any, Literal, TypeAlias, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator


SubAgentName = Literal["paper_search_agent"]

RunStatus: TypeAlias = Literal[
    "running",
    "waiting_confirmation",
    "completed",
    "failed",
    "blocked",
]


class ToolRequestState(TypedDict):
    name: str
    arguments: dict[str, Any]


class SubAgentRequestState(TypedDict):
    name: SubAgentName
    input: dict[str, Any]


class SolveDecisionState(TypedDict):
    action: Literal["direct", "tool", "subagent"]
    answer: str | None
    tool_name: str | None
    tool_arguments: dict[str, Any]
    subagent_name: SubAgentName | None
    subagent_input: dict[str, Any]
    confirmation_hint: str | None
    decision_reason: str


class PendingActionState(TypedDict):
    action_id: str
    action: Literal["tool", "subagent"]
    tool_name: str | None
    tool_arguments: dict[str, Any]
    subagent_name: SubAgentName | None
    subagent_input: dict[str, Any]
    confirmation_hint: str | None
    decision_reason: str
    requires_confirmation: bool
    confirmation_message: str | None


class ConfirmationRecordState(TypedDict):
    action_id: str
    decision: Literal["approved", "rejected"]
    comment: str | None


class ActionResultState(TypedDict):
    action_id: str
    action_type: Literal["tool", "subagent"]
    name: str
    status: Literal["success", "accepted", "partial", "rejected", "error"]
    summary: str
    data: dict[str, Any]
    artifact_refs: list[str]
    retryable: bool
    error_code: str | None
    error_message: str | None


class ActionExecutionRecordState(TypedDict):
    action_id: str
    action_type: Literal["tool", "subagent"]
    name: str
    arguments_digest: str
    status: str
    summary: str


class SolveDecision(BaseModel):
    action: Literal["direct", "tool", "subagent"]
    answer: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    subagent_name: SubAgentName | None = None
    subagent_input: dict[str, Any] = Field(default_factory=dict)
    confirmation_hint: str | None = None
    decision_reason: str = ""

    @field_validator("answer", "tool_name", "subagent_name", mode="before")
    @classmethod
    def normalize_nullable_string(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
            return None
        return value

    @field_validator("tool_arguments", "subagent_input", mode="before")
    @classmethod
    def normalize_nullable_dict(cls, value: Any) -> dict[str, Any]:
        if value is None or (
            isinstance(value, str) and value.strip().lower() in {"none", "null", ""}
        ):
            return {}
        return value

    @model_validator(mode="after")
    def validate_action_payload(self) -> "SolveDecision":
        if self.action == "tool" and not self.tool_name:
            raise ValueError("action=tool requires tool_name")
        if self.action == "subagent" and self.subagent_name is None:
            raise ValueError("action=subagent requires subagent_name")
        return self

    def to_state(self) -> SolveDecisionState:
        return {
            "action": self.action,
            "answer": self.answer,
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments,
            "subagent_name": self.subagent_name,
            "subagent_input": self.subagent_input,
            "confirmation_hint": self.confirmation_hint,
            "decision_reason": self.decision_reason,
        }
