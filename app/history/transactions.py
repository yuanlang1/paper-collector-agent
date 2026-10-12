from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.history import chat_log


def persist_message_event(
    db: Session,
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    event_name: str,
    event_data: dict[str, Any],
    content: str | None = None,
) -> None:
    if content:
        chat_log.append_text_message(
            db,
            user_id=user_id,
            conversation_id=conversation_id,
            run_id=run_id,
            content=content,
        )
    if event_name == "action_started":
        batch_id, batch_calls = _batch(event_data, run_id=run_id)
        chat_log.append_tool_batch(
            db,
            user_id=user_id,
            conversation_id=conversation_id,
            run_id=run_id,
            batch_id=batch_id,
            batch_calls=batch_calls,
        )
    elif event_name == "action_result":
        batch_id, batch_calls = _batch(event_data, run_id=run_id)
        chat_log.append_tool_result(
            db,
            user_id=user_id,
            conversation_id=conversation_id,
            run_id=run_id,
            batch_id=batch_id,
            batch_calls=batch_calls,
            result=event_data,
        )


def complete_assistant_message(
    db: Session,
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    response: dict[str, Any],
    latency_ms: int,
    extra_meta: dict[str, Any] | None,
    content: str | None = None,
) -> int | None:
    return chat_log.complete_run(
        db,
        user_id=user_id,
        conversation_id=conversation_id,
        run_id=run_id,
        response=response,
        latency_ms=latency_ms,
        extra_meta=extra_meta,
        content=content,
    )


def _batch(event_data: dict[str, Any], *, run_id: str) -> tuple[str, list[dict[str, Any]]]:
    action_id = str(event_data.get("action_id") or "")
    calls = event_data.get("batch_calls")
    if isinstance(calls, list) and all(isinstance(item, dict) for item in calls):
        normalized = [dict(item) for item in calls]
    else:
        normalized = [{
            "id": action_id,
            "name": str(event_data.get("name") or ""),
            "args": event_data.get("input") or {},
            "kind": event_data.get("action_type") or "tool",
            "requires_confirmation": bool(event_data.get("requires_confirmation")),
        }]
    if not action_id:
        raise ValueError("action event must include action_id")
    if not any(str(item.get("id") or "") == action_id for item in normalized):
        raise ValueError("action event is absent from its tool batch")
    return str(event_data.get("batch_id") or f"{run_id}:legacy:{action_id}"), normalized
