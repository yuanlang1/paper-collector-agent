from __future__ import annotations

from typing import Any

from app.history import chat_log, stream_events
from app.history.sqlite import HistoryDatabase
from app.history.stream_events import AgentStreamEvent


def complete_assistant_message_with_event(
    database: HistoryDatabase,
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    message_id: int,
    response: dict[str, Any],
    latency_ms: int,
    extra_meta: dict[str, Any] | None,
    event_name: str,
    event_data: dict[str, Any],
) -> AgentStreamEvent:
    with database.transaction() as connection:
        chat_log.complete_assistant_message_in_transaction(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
            run_id=run_id,
            message_id=message_id,
            response=response,
            latency_ms=latency_ms,
            extra_meta=extra_meta,
        )
        return stream_events.insert(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
            run_id=run_id,
            event_name=event_name,
            data=event_data,
        )


def delete_conversation(
    database: HistoryDatabase,
    *,
    user_id: str,
    conversation_id: str,
) -> int:
    with database.transaction() as connection:
        stream_events.delete_conversation(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        return chat_log.delete_conversation(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
        )
