from __future__ import annotations

from typing import Literal

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
    max_reflection_rounds: int = Field(default=5, ge=1, le=5)
