from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, TypedDict

from pydantic import BaseModel, Field


MemoryStatus = Literal[
    "active",
    "pending",
    "rejected",
    "superseded",
    "deleted",
]

MemoryRetrievalStatus = Literal[
    "skipped",
    "running",
    "completed",
    "empty",
    "failed",
]


class MemoryUsage(TypedDict):
    status: MemoryRetrievalStatus
    facts_count: int
    episodes_count: int


@dataclass(frozen=True)
class MemoryContextResult:
    content: str
    usage: MemoryUsage


@dataclass(frozen=True)
class FactCandidate:
    subject: str
    content: str
    confidence: float = 0.8


@dataclass(frozen=True)
class EpisodeCandidate:
    happened_at: date
    summary: str
    source_assistant_message_id: int


@dataclass(frozen=True)
class HistoryTurn:
    user_message_id: int
    assistant_message_id: int
    user_content: str
    assistant_content: str
    completed_at: datetime


@dataclass(frozen=True)
class ConsolidationOutput:
    facts: list[FactCandidate]
    episode: EpisodeCandidate | None


@dataclass(frozen=True)
class ConsolidationResult:
    due: bool
    facts_created: int
    episode_created: bool
    last_assistant_message_id: int | None


class ExtractedFact(BaseModel):
    """The LLM-facing fact schema used during consolidation."""

    subject: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=1000)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class MemoryExtractionPayload(BaseModel):
    """Structured output from the consolidation model.

    Source message IDs and dates deliberately stay out of this schema: they
    are assigned by the application from the real chat-history rows.
    """

    facts: list[ExtractedFact] = Field(default_factory=list, max_length=3)
    episode_summary: str | None = Field(default=None, max_length=1000)
