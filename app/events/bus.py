from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel

from app.events.adapter import sse_name_for
from app.events.delivery import EventContext, EventDelivery
from app.events.models import Event, EventBase

if TYPE_CHECKING:
    from app.events.handlers.ipc_event_broadcaster import IpcEventBroadcaster
    from app.events.handlers.persistence import PersistenceHandler


EventHandler = Callable[[BaseModel], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> None:
        self._subscribers.append(handler)

    async def publish(self, event: BaseModel) -> None:
        for handler in self._subscribers:
            await handler(event)


class RootEventDispatcher:
    def __init__(
        self,
        *,
        context: EventContext,
        persistence: PersistenceHandler,
        broadcaster: IpcEventBroadcaster,
    ) -> None:
        self._context = context
        self._persistence = persistence
        self._broadcaster = broadcaster
        self._lock = asyncio.Lock()

    async def __call__(self, event: BaseModel) -> None:
        if not isinstance(event, EventBase):
            raise TypeError(f"Expected an event model, got {type(event).__name__}")
        typed_event: Event = event
        delivery = EventDelivery(
            context=self._context,
            event=typed_event,
            sse_name=sse_name_for(typed_event),
        )
        async with self._lock:
            await self._persistence(delivery)
            await self._broadcaster(delivery)
