from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Mapping

from app.events.models import (
    ActionResultEvent,
    ActionStartedEvent,
    ContentDeltaEvent,
    Event,
    IterationStartedEvent,
    MemoryProgressEvent,
    NodeErrorEvent,
    ReasoningDeltaEvent,
    RunStartedEvent,
    RunTerminalEvent,
    SubagentFinishedEvent,
    SubagentProgressEvent,
    SubagentStartedEvent,
    TimelineStepEvent,
    ToolFailedEvent,
    ToolFinishedEvent,
    ToolStartedEvent,
)
from app.events.envelope import EventEnvelope, build_event
from app.llm.streaming.utils import to_jsonable


def now() -> str:
    return datetime.now(UTC).isoformat()


def event_from_envelope(envelope: Mapping[str, Any], *, run_id: str) -> Event:
    name = str(envelope["event"])
    payload = {
        key: to_jsonable(value)
        for key, value in envelope.items()
        if key not in {"event", "run_id", "conversation_id"}
    }
    timestamp = now()

    if name == "run_started":
        return RunStartedEvent(run_id=run_id, ts=timestamp, payload=payload)
    if name in {"run_completed", "run_failed", "confirmation_required"}:
        status = str(payload["status"])
        event_type = (
            "run.waiting_for_confirmation"
            if name == "confirmation_required"
            else "run.finished"
        )
        return RunTerminalEvent(
            type=event_type,
            run_id=run_id,
            ts=timestamp,
            status=status,
            payload=payload,
        )

    event_class = {
        "iteration_started": IterationStartedEvent,
        "content_delta": ContentDeltaEvent,
        "message": ContentDeltaEvent,
        "reasoning_delta": ReasoningDeltaEvent,
        "action_started": ActionStartedEvent,
        "action_result": ActionResultEvent,
        "tool_started": ToolStartedEvent,
        "tool_completed": ToolFinishedEvent,
        "tool_failed": ToolFailedEvent,
        "subagent_started": SubagentProgressEvent,
        "subagent_progress": SubagentProgressEvent,
        "subagent_completed": SubagentProgressEvent,
        "subagent_failed": SubagentProgressEvent,
        "timeline_step": TimelineStepEvent,
        "node_error": NodeErrorEvent,
    }.get(name)
    if event_class is not None:
        if event_class is SubagentProgressEvent:
            payload["sse_name"] = name
        return event_class(run_id=run_id, ts=timestamp, payload=payload)
    if name.startswith("memory_"):
        payload["sse_name"] = name
        return MemoryProgressEvent(run_id=run_id, ts=timestamp, payload=payload)
    raise ValueError(f"Unsupported stream event: {name}")


def sse_name_for(event: Event) -> str:
    if isinstance(event, RunStartedEvent):
        return "run_started"
    if isinstance(event, RunTerminalEvent):
        if event.status == "confirmation_required":
            return "confirmation_required"
        return "run_failed" if event.status == "failed" else "run_completed"
    if isinstance(event, IterationStartedEvent):
        return "iteration_started"
    if isinstance(event, ContentDeltaEvent):
        return "message" if "content" in event.payload else "content_delta"
    if isinstance(event, ReasoningDeltaEvent):
        return "reasoning_delta"
    if isinstance(event, ActionStartedEvent):
        return "action_started"
    if isinstance(event, ActionResultEvent):
        return "action_result"
    if isinstance(event, ToolStartedEvent):
        return "tool_started"
    if isinstance(event, ToolFinishedEvent):
        return "tool_completed"
    if isinstance(event, ToolFailedEvent):
        return "tool_failed"
    if isinstance(event, SubagentStartedEvent):
        return "subagent_started"
    if isinstance(event, SubagentFinishedEvent):
        return "subagent_completed" if event.status in {"success", "partial"} else "subagent_failed"
    if isinstance(event, SubagentProgressEvent):
        return str(event.payload["sse_name"])
    if isinstance(event, TimelineStepEvent):
        return "timeline_step"
    if isinstance(event, MemoryProgressEvent):
        return str(event.payload["sse_name"])
    if isinstance(event, NodeErrorEvent):
        return "node_error"
    raise TypeError(f"Unsupported event type: {type(event).__name__}")


def event_data(event: Event) -> dict[str, Any]:
    if isinstance(event, (SubagentStartedEvent, SubagentFinishedEvent)):
        data = event.model_dump(exclude={"run_id"})
        data["run_id"] = event.run_id
        return data

    data = {
        "type": event.type,
        "ts": event.ts,
        "run_id": event.run_id,
        **{key: value for key, value in event.payload.items() if key != "sse_name"},
    }
    if isinstance(event, RunTerminalEvent):
        data["status"] = event.status
    return data
