from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class TaskReviewDelegation(BaseModel):
    task_id: int = Field(gt=0)
    topic: str = Field(min_length=2, max_length=2_000)
    language: Literal["zh-CN", "en"] = "zh-CN"
    citation_style: Literal[
        "harvard",
        "apa",
        "ieee",
        "chicago",
        "vancouver",
    ] = "harvard"
    review_type: Literal[
        "narrative",
        "systematic",
        "scoping",
        "critical",
    ] = "narrative"
    max_reflection_rounds: int = Field(default=2, ge=1, le=3)


class TaskReviewHandoff(BaseModel):
    status: Literal["success", "partial", "error", "rejected"]
    summary: str
    data: dict[str, Any]
    artifact_refs: list[str]
    retryable: bool
    error_code: str | None = None
    error_message: str | None = None


class TaskReviewRequestState(TypedDict):
    task_id: int
    topic: str
    language: str
    citation_style: str
    review_type: str
    max_reflection_rounds: int


class TaskReviewHandoffState(TypedDict):
    status: Literal["success", "partial", "error", "rejected"]
    summary: str
    data: dict[str, Any]
    artifact_refs: list[str]
    retryable: bool
    error_code: str | None
    error_message: str | None
