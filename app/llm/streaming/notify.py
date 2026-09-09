from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer


EventEnvelope = dict[str, Any]
EventSink = Callable[[EventEnvelope], None]


def build_event(
    kind: str,
    payload: Mapping[str, Any] | None = None,
    scope: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    return {
        **dict(payload or {}),
        **dict(scope or {}),
        "event": kind,
    }


@dataclass(frozen=True)
class Notifier:
    sink: EventSink
    scope: Mapping[str, Any] = field(default_factory=dict)

    def __call__(
        self,
        kind: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        self.sink(build_event(kind, payload, self.scope))

    def scoped(self, **scope: Any) -> "Notifier":
        return Notifier(self.sink, {**self.scope, **scope})


def _noop_sink(_: EventEnvelope) -> None:
    return None


NOOP_NOTIFIER = Notifier(_noop_sink)


def langgraph_notifier(config: RunnableConfig | None = None) -> Notifier:
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return NOOP_NOTIFIER

    metadata = config.get("metadata", {}) if config else {}
    scope = metadata.get("notify_scope", {}) if isinstance(metadata, Mapping) else {}
    return Notifier(writer, dict(scope) if isinstance(scope, Mapping) else {})
