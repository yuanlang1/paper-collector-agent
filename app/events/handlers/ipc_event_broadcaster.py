from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from app.events.delivery import EventDelivery
from app.history import chat_log, stream_events
from app.history.sqlite import HistoryDatabase
from app.history.stream_events import AgentStreamEvent

class IpcEventBroadcaster:
    def __init__(self, history_db: HistoryDatabase) -> None:
        self._history_db = history_db
        self._subscribers: dict[tuple[str, str], set[asyncio.Event]] = {}

    async def __call__(self, delivery: EventDelivery) -> None:
        if delivery.persisted is None:
            raise RuntimeError("Cannot broadcast an event before it is persisted")
        key = (delivery.context.user_id, delivery.context.root_run_id)
        for notification in self._subscribers.get(key, set()).copy():
            notification.set()

    async def iter_events(
        self,
        *,
        user_id: str,
        root_run_id: str,
        after_sequence: int | None,
    ) -> AsyncIterator[AgentStreamEvent]:
        key = (user_id, root_run_id)
        notification = asyncio.Event()
        self._subscribers.setdefault(key, set()).add(notification)
        cursor = after_sequence
        try:
            while True:
                notification.clear()
                status = await self._history_db.run(
                    chat_log.get_run_status,
                    user_id=user_id,
                    run_id=root_run_id,
                )
                events = await self._history_db.run(
                    stream_events.list_after,
                    user_id=user_id,
                    run_id=root_run_id,
                    after_sequence=cursor,
                    limit=100,
                )
                for event in events:
                    cursor = event.sequence
                    yield event
                if events:
                    continue
                if status in {
                    "completed",
                    "confirmation_required",
                    "failed",
                    "blocked",
                    "interrupted",
                }:
                    return
                try:
                    await asyncio.wait_for(notification.wait(), timeout=2)
                except TimeoutError:
                    pass
        finally:
            subscribers = self._subscribers.get(key)
            if subscribers is not None:
                subscribers.discard(notification)
                if not subscribers:
                    self._subscribers.pop(key, None)
