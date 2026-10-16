from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel

EventHandler = Callable[[BaseModel], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> None:
        self._subscribers.append(handler)

    async def publish(self, event: BaseModel) -> None:
        for handler in self._subscribers:
            try:
                await handler(event)
            except Exception as exc:
                pass