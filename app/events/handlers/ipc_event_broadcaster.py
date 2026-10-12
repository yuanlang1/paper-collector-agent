from __future__ import annotations

import asyncio
from time import monotonic
from collections.abc import AsyncIterator

from app.events.delivery import EventDelivery
from app.history import chat_log, stream_events
from app.history.chat_log import AgentHistoryStore
from app.history.sqlite import HistoryDatabase
from app.history.stream_events import AgentStreamEvent

class IpcEventBroadcaster:
    def __init__(self, event_db: HistoryDatabase, history_store: AgentHistoryStore) -> None:
        self._event_db = event_db
        self._history_store = history_store
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
        terminal_wait_started_at: float | None = None
        try:
            while True:
                notification.clear()
                run_state = await self._history_store.run(
                    chat_log.get_run_event_state,
                    user_id=user_id,
                    run_id=root_run_id,
                )
                events = await self._event_db.run(
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
                if run_state is not None and run_state.status in {
                    "completed",
                    "confirmation_required",
                    "failed",
                    "blocked",
                    "interrupted",
                }:
                    expected = run_state.terminal_event_sequence
                    if expected is not None and cursor is not None and cursor >= expected:
                        return
                    terminal_wait_started_at = terminal_wait_started_at or monotonic()
                    if monotonic() - terminal_wait_started_at >= 10:
                        raise RuntimeError("terminal event was not persisted")
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
