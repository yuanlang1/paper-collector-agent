from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, TypeAdapter

from app.events.delivery import TerminalUpdate


class EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    ts: str


class PayloadEvent(EventBase):
    payload: dict[str, Any] = Field(default_factory=dict)


class RunStartedEvent(PayloadEvent):
    type: Literal["run.started"] = "run.started"


class RunTerminalEvent(PayloadEvent):
    type: Literal["run.finished", "run.waiting_for_confirmation"]
    status: Literal["completed", "confirmation_required", "failed", "blocked"]
    terminal: TerminalUpdate | None = Field(default=None, exclude=True)


class IterationStartedEvent(PayloadEvent):
    type: Literal["iteration.started"] = "iteration.started"


class ContentDeltaEvent(PayloadEvent):
    type: Literal["content.delta"] = "content.delta"


class ReasoningDeltaEvent(PayloadEvent):
    type: Literal["reasoning.delta"] = "reasoning.delta"


class ActionStartedEvent(PayloadEvent):
    type: Literal["action.started"] = "action.started"


class ActionResultEvent(PayloadEvent):
    type: Literal["action.result"] = "action.result"


class ToolStartedEvent(PayloadEvent):
    type: Literal["tool.started"] = "tool.started"


class ToolFinishedEvent(PayloadEvent):
    type: Literal["tool.finished"] = "tool.finished"


class ToolFailedEvent(PayloadEvent):
    type: Literal["tool.failed"] = "tool.failed"


class SubagentProgressEvent(PayloadEvent):
    type: Literal["subagent.progress"] = "subagent.progress"


class TimelineStepEvent(PayloadEvent):
    type: Literal["timeline.step"] = "timeline.step"


class MemoryProgressEvent(PayloadEvent):
    type: Literal["memory.progress"] = "memory.progress"


class NodeErrorEvent(PayloadEvent):
    type: Literal["node.error"] = "node.error"


class SubagentStartedEvent(EventBase):
    type: Literal["subagent.started"] = "subagent.started"
    parent_run_id: str
    description: str


class SubagentFinishedEvent(EventBase):
    type: Literal["subagent.finished"] = "subagent.finished"
    parent_run_id: str
    status: Literal["success", "partial", "failed", "cancelled"]


Event = Annotated[
    RunStartedEvent
    | RunTerminalEvent
    | IterationStartedEvent
    | ContentDeltaEvent
    | ReasoningDeltaEvent
    | ActionStartedEvent
    | ActionResultEvent
    | ToolStartedEvent
    | ToolFinishedEvent
    | ToolFailedEvent
    | SubagentProgressEvent
    | TimelineStepEvent
    | MemoryProgressEvent
    | NodeErrorEvent
    | SubagentStartedEvent
    | SubagentFinishedEvent,
    Discriminator("type"),
]

EVENT_ADAPTER = TypeAdapter(Event)
