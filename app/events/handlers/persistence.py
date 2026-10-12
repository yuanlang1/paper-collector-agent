from __future__ import annotations

from time import perf_counter

from app.events.adapter import event_data
from app.events.delivery import EventDelivery
from app.events.models import ContentDeltaEvent, RunTerminalEvent
from app.history import chat_log, transactions
from app.history.chat_log import AgentHistoryStore
from app.history.sqlite import HistoryDatabase


class PersistenceHandler:
    def __init__(
        self,
        history_store: AgentHistoryStore,
        event_db: HistoryDatabase,
        *,
        started_at: float,
    ) -> None:
        self._history_store = history_store
        self._event_db = event_db
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
            await self._history_store.run(
                transactions.persist_message_event,
                user_id=context.user_id,
                conversation_id=context.conversation_id,
                run_id=context.root_run_id,
                event_name=delivery.sse_name,
                event_data=data,
                content=pending_content,
            )
            delivery.persisted = await self._event_db.run(
                _append_event,
                user_id=context.user_id,
                conversation_id=context.conversation_id,
                run_id=context.root_run_id,
                event_name=delivery.sse_name,
                event_data=data,
            )
            if pending_content:
                self._content_parts.clear()
            return

        pending_content = self._pending_content()
        await self._history_store.run(
            transactions.complete_assistant_message,
            user_id=context.user_id,
            conversation_id=context.conversation_id,
            run_id=context.root_run_id,
            response=terminal.response,
            latency_ms=int((perf_counter() - self._started_at) * 1000),
            extra_meta=terminal.extra_meta,
            content=pending_content,
        )
        delivery.persisted = await self._event_db.run(
            _append_event,
            user_id=context.user_id,
            conversation_id=context.conversation_id,
            run_id=context.root_run_id,
            event_name=delivery.sse_name,
            event_data=data,
        )
        await self._history_store.run(
            chat_log.record_terminal_event_sequence,
            user_id=context.user_id,
            run_id=context.root_run_id,
            sequence=delivery.persisted.sequence,
        )
        if pending_content:
            self._content_parts.clear()

    def _pending_content(self) -> str | None:
        return "".join(self._content_parts) or None


def _append_event(
    database: HistoryDatabase,
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    event_name: str,
    event_data: dict,
):
    from app.history import stream_events

    return stream_events.append(
        database,
        user_id=user_id,
        conversation_id=conversation_id,
        run_id=run_id,
        event_name=event_name,
        data=event_data,
    )
