from __future__ import annotations

from time import perf_counter

from app.events.adapter import event_data
from app.events.delivery import EventDelivery
from app.events.models import ContentDeltaEvent, RunTerminalEvent
from app.history import transactions
from app.history.sqlite import HistoryDatabase


class PersistenceHandler:
    def __init__(self, history_db: HistoryDatabase, *, started_at: float) -> None:
        self._history_db = history_db
        self._started_at = started_at
        self._content_parts: list[str] = []

    async def __call__(self, delivery: EventDelivery) -> None:
        context = delivery.context
        data = event_data(delivery.event)
        terminal = (
            delivery.event.terminal
            if isinstance(delivery.event, RunTerminalEvent)
            else None
        )
        if terminal is None:
            if isinstance(delivery.event, ContentDeltaEvent):
                value = data.get("delta") or data.get("content")
                if isinstance(value, str):
                    self._content_parts.append(value)
            pending_content = (
                self._pending_content()
                if delivery.sse_name == "action_started"
                else None
            )
            delivery.persisted = await self._history_db.run(
                transactions.persist_stream_event,
                user_id=context.user_id,
                conversation_id=context.conversation_id,
                run_id=context.root_run_id,
                event_name=delivery.sse_name,
                event_data=data,
                content=pending_content,
            )
            if pending_content:
                self._content_parts.clear()
            return

        pending_content = self._pending_content()
        delivery.persisted = await self._history_db.run(
            transactions.complete_assistant_message_with_event,
            user_id=context.user_id,
            conversation_id=context.conversation_id,
            run_id=context.root_run_id,
            response=terminal.response,
            latency_ms=int((perf_counter() - self._started_at) * 1000),
            extra_meta=terminal.extra_meta,
            event_name=delivery.sse_name,
            event_data=data,
            content=pending_content,
        )
        if pending_content:
            self._content_parts.clear()

    def _pending_content(self) -> str | None:
        return "".join(self._content_parts) or None
