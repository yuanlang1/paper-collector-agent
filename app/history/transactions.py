from __future__ import annotations

from typing import Any

from app.history import chat_log, stream_events
from app.history.sqlite import HistoryDatabase
from app.history.stream_events import AgentStreamEvent


def persist_stream_event(
    database: HistoryDatabase,
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    event_name: str,
    event_data: dict[str, Any],
    content: str | None = None,
) -> AgentStreamEvent:
    with database.transaction() as connection:
        if content:
            chat_log.append_message_in_transaction(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                run_id=run_id,
                role="assistant",
                content=[{"type": "text", "text": content}],
                status="completed",
            )
        if event_name == "action_started":
            chat_log.append_action_started_in_transaction(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                run_id=run_id,
                action=event_data,
            )
        elif event_name == "action_result":
            chat_log.append_action_result_in_transaction(
                connection,
                user_id=user_id,
                conversation_id=conversation_id,
                run_id=run_id,
                result=event_data,
            )
        return stream_events.insert(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
            run_id=run_id,
            event_name=event_name,
            data=event_data,
        )


def complete_assistant_message_with_event(
    database: HistoryDatabase,
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    response: dict[str, Any],
    latency_ms: int,
    extra_meta: dict[str, Any] | None,
    event_name: str,
    event_data: dict[str, Any],
    content: str | None = None,
) -> AgentStreamEvent:
    with database.transaction() as connection:
        chat_log.append_terminal_assistant_message_in_transaction(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
            run_id=run_id,
            response=response,
            latency_ms=latency_ms,
            extra_meta=extra_meta,
            content=content,
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
