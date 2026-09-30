from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.history.stream_events import AgentStreamEvent

if TYPE_CHECKING:
    from app.events.models import Event


@dataclass(frozen=True)
class EventContext:
    user_id: str
    conversation_id: str
    root_run_id: str
    assistant_message_id: int


@dataclass
class EventDelivery:
    context: EventContext
    event: "Event"
    sse_name: str
    persisted: AgentStreamEvent | None = None


@dataclass(frozen=True)
class TerminalUpdate:
    response: dict[str, Any]
    card_meta: dict[str, Any]
