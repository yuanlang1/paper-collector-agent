from __future__ import annotations

from typing import Any, Mapping


EventEnvelope = dict[str, Any]


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
