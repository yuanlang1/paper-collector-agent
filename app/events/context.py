from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from app.events.adapter import event_from_envelope
from app.events.bus import EventBus
from app.events.envelope import build_event
from app.events.errors import EventPublicationError
from app.events.models import Event


@dataclass(frozen=True)
class EventContext:
    bus: EventBus
    run_id: str
    scope: Mapping[str, Any] = field(default_factory=dict)

    def scoped(self, **scope: Any) -> "EventContext":
        return EventContext(
            bus=self.bus,
            run_id=self.run_id,
            scope={**self.scope, **scope},
        )

    def with_bus(self, bus: EventBus) -> "EventContext":
        return EventContext(bus=bus, run_id=self.run_id, scope=self.scope)

    def event(
        self,
        name: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Event:
        return event_from_envelope(
            build_event(name, payload, self.scope),
            run_id=self.run_id,
        )


_event_context: ContextVar[EventContext | None] = ContextVar(
    "event_context",
    default=None,
)


@contextmanager
def bind_event_context(context: EventContext) -> Iterator[EventContext]:
    token = _event_context.set(context)
    try:
        yield context
    finally:
        _event_context.reset(token)


def current_event_context() -> EventContext:
    context = _event_context.get()
    if context is None:
        raise EventPublicationError("No event context is bound to this execution.")
    return context
