from __future__ import annotations

from time import perf_counter

from app.events.adapter import event_data
from app.events.delivery import EventDelivery
from app.events.models import RunTerminalEvent
from app.history import stream_events, transactions
from app.history.sqlite import HistoryDatabase


class PersistenceHandler:
    def __init__(self, history_db: HistoryDatabase, *, started_at: float) -> None:
        self._history_db = history_db
        self._started_at = started_at

    async def __call__(self, delivery: EventDelivery) -> None:
        context = delivery.context
        data = event_data(delivery.event)
        terminal = (
            delivery.event.terminal
            if isinstance(delivery.event, RunTerminalEvent)
            else None
        )
        if terminal is None:
            delivery.persisted = await self._history_db.run(
                stream_events.append,
                user_id=context.user_id,
                conversation_id=context.conversation_id,
                run_id=context.root_run_id,
                event_name=delivery.sse_name,
                data=data,
            )
            return

        delivery.persisted = await self._history_db.run(
            transactions.complete_assistant_message_with_event,
            user_id=context.user_id,
            conversation_id=context.conversation_id,
            run_id=context.root_run_id,
            message_id=context.assistant_message_id,
            response=terminal.response,
            latency_ms=int((perf_counter() - self._started_at) * 1000),
            extra_meta=terminal.card_meta,
            event_name=delivery.sse_name,
            event_data=data,
        )
