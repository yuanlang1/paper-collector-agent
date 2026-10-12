from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, TypeVar

from sqlalchemy import Select, delete, select
from sqlalchemy.orm import Session, sessionmaker

from app.memory.schemas import HistoryTurn
from app.models.agent_history import AgentConversation, AgentMessage, AgentRun


T = TypeVar("T")
TERMINAL_RUN_STATUSES = {"SUCCEEDED", "FAILED", "CANCELLED"}
ACTIVE_RUN_STATUSES = {"PENDING", "RUNNING", "WAITING_INPUT"}
_CANONICAL_RUN_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,62}[A-Za-z0-9])?")


class PendingActionConflictError(ValueError):
    """The requested approval cannot be applied to the stored tool call."""


class InvalidRunIdError(ValueError):
    """The run ID cannot be persisted as a stable agent-run identifier."""


class RunAlreadyExistsError(ValueError):
    """The requested root run has already been created."""


class ActiveRunConflictError(ValueError):
    """Another unfinished run already owns the conversation thread."""


class ConversationNotFoundError(LookupError):
    """The conversation is absent or is owned by another user."""


class ConversationStateError(ValueError):
    """The conversation cannot accept a new agent run."""


class ConversationThreadError(ValueError):
    """The conversation does not have a LangGraph thread ID."""


class ToolResultConflictError(ValueError):
    """A tool result was retried with a different payload."""


@dataclass(frozen=True)
class RunScope:
    user_id: str
    conversation_id: str


@dataclass(frozen=True)
class RunView:
    run_id: str
    conversation_id: str
    status: str
    reply: str
    result: dict[str, Any]
    terminal_event_sequence: int | None


@dataclass(frozen=True)
class RunEventState:
    status: str
    terminal_event_sequence: int | None


class AgentHistoryStore:
    """Runs short, independent SQLAlchemy transactions for agent history."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    async def run(self, operation: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        return await asyncio.to_thread(self._run, operation, *args, **kwargs)

    def _run(self, operation: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        db = self._session_factory()
        try:
            result = operation(db, *args, **kwargs)
            db.commit()
            return result
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()


def validate_run_id(value: str) -> str:
    if not _CANONICAL_RUN_ID.fullmatch(value):
        raise InvalidRunIdError(
            "run_id must be 1-64 ASCII letters, digits, '.', '_' or '-', "
            "and must start and end with a letter or digit"
        )
    return value


def validate_conversation_id(value: str) -> str:
    value = value.strip()
    if not value or len(value) > 64:
        raise ValueError("conversation_id must be 1-64 characters")
    return value


def normalize_user_id(value: str | int) -> tuple[str, int]:
    raw = str(value).strip()
    if not raw.isdecimal():
        raise ValueError("user_id must be a non-negative BIGINT")
    numeric = int(raw)
    if numeric > 9_223_372_036_854_775_807:
        raise ValueError("user_id must be a non-negative BIGINT")
    return str(numeric), numeric


def start_turn(
    db: Session,
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    user_content: str,
    source: str = "api",
) -> str:
    _, user_number = normalize_user_id(user_id)
    conversation_id = validate_conversation_id(conversation_id)
    run_id = validate_run_id(run_id)
    conversation = _locked_conversation(db, user_number=user_number, conversation_id=conversation_id)
    _require_runnable_conversation(conversation)
    if db.scalar(select(AgentRun.id).where(AgentRun.run_id == run_id).with_for_update()) is not None:
        raise RunAlreadyExistsError("run_id already exists")
    active_run = db.scalar(
        select(AgentRun.id).where(
            AgentRun.creator_id == user_number,
            AgentRun.conversation_id == conversation_id,
            AgentRun.status.in_(ACTIVE_RUN_STATUSES),
        ).with_for_update()
    )
    if active_run is not None:
        raise ActiveRunConflictError("conversation already has an active run")
    user_message = _add_message(
        db, user_number=user_number, conversation_id=conversation_id, run_id=run_id,
        role="user", blocks=[{"type": "text", "text": user_content}],
        status="COMPLETED", payload={"source": source, "kind": "input"},
    )
    db.flush()
    db.add(AgentRun(
        creator_id=user_number, modifier_id=user_number, run_id=run_id,
        conversation_id=conversation_id, status="RUNNING", input_message_id=user_message.id,
        creation_time=_utc_datetime(), modification_time=_utc_datetime(),
    ))
    return _thread_id(conversation)


def get_conversation_binding(db: Session, *, user_id: str, conversation_id: str) -> str:
    _, user_number = normalize_user_id(user_id)
    conversation = _conversation_for_user(
        db, user_number=user_number, conversation_id=validate_conversation_id(conversation_id)
    )
    if conversation is None:
        raise ConversationNotFoundError("conversation not found")
    _require_runnable_conversation(conversation)
    return _thread_id(conversation)


def append_text_message(
    db: Session, *, user_id: str, conversation_id: str, run_id: str, content: str
) -> int:
    _, user_number = normalize_user_id(user_id)
    _lock_run(db, user_number=user_number, conversation_id=conversation_id, run_id=run_id)
    message = _add_message(
        db, user_number=user_number, conversation_id=conversation_id, run_id=run_id,
        role="assistant", blocks=[{"type": "text", "text": content}], status="COMPLETED",
        payload={"kind": "text"},
    )
    db.flush()
    return int(message.id)


def append_tool_batch(
    db: Session, *, user_id: str, conversation_id: str, run_id: str,
    batch_id: str, batch_calls: list[dict[str, Any]],
) -> int:
    _, user_number = normalize_user_id(user_id)
    _lock_run(db, user_number=user_number, conversation_id=conversation_id, run_id=run_id)
    existing = _find_batch_message(db, run_id=run_id, role="assistant", batch_id=batch_id, kind="tool_calls")
    blocks = _tool_call_blocks(batch_calls)
    call_manifest = _tool_call_manifest(batch_calls)
    if existing is not None:
        if (
            decode_content(existing.content) != blocks
            or _json_dict(existing.payload).get("tool_calls") != call_manifest
        ):
            raise ToolResultConflictError("tool batch was retried with different calls")
        return int(existing.id)
    message = _add_message(
        db, user_number=user_number, conversation_id=conversation_id, run_id=run_id,
        role="assistant", blocks=blocks, status="COMPLETED",
        payload={"kind": "tool_calls", "batch_id": batch_id,
                 "tool_call_ids": [str(call["id"]) for call in batch_calls],
                 "tool_calls": call_manifest},
    )
    db.flush()
    return int(message.id)


def append_tool_result(
    db: Session, *, user_id: str, conversation_id: str, run_id: str, batch_id: str,
    batch_calls: list[dict[str, Any]], result: dict[str, Any],
) -> int:
    _, user_number = normalize_user_id(user_id)
    _lock_run(db, user_number=user_number, conversation_id=conversation_id, run_id=run_id)
    call_ids = [str(call["id"]) for call in batch_calls]
    call_manifest = _tool_call_manifest(batch_calls)
    result_block = _tool_result_block(result)
    tool_use_id = str(result_block["tool_use_id"])
    if tool_use_id not in call_ids:
        raise ValueError("tool result does not belong to its declared batch")
    message = _find_batch_message(db, run_id=run_id, role="user", batch_id=batch_id, kind="tool_results")
    if message is None:
        blocks: list[dict[str, Any]] = []
        message = _add_message(
            db, user_number=user_number, conversation_id=conversation_id, run_id=run_id,
            role="user", blocks=blocks, status="STREAMING",
            payload={"kind": "tool_results", "batch_id": batch_id,
                     "tool_call_ids": call_ids, "tool_calls": call_manifest},
        )
        db.flush()
    else:
        if _json_dict(message.payload).get("tool_calls") != call_manifest:
            raise ToolResultConflictError("tool result batch was retried with different calls")
        blocks = _blocks_or_empty(message.content)
    existing_by_id = {
        str(block.get("tool_use_id")): block
        for block in blocks if block.get("type") == "tool_result"
    }
    prior = existing_by_id.get(tool_use_id)
    if prior is not None:
        if prior != result_block:
            raise ToolResultConflictError("tool result was retried with different content")
        return int(message.id)
    existing_by_id[tool_use_id] = result_block
    ordered_blocks = [existing_by_id[call_id] for call_id in call_ids if call_id in existing_by_id]
    message.content = _serialize(ordered_blocks)
    message.status = "COMPLETED" if len(ordered_blocks) == len(call_ids) else "STREAMING"
    message.modifier_id = user_number
    message.modification_time = _utc_datetime()
    return int(message.id)


def complete_run(
    db: Session, *, user_id: str, conversation_id: str, run_id: str, response: dict[str, Any],
    latency_ms: int, extra_meta: dict[str, Any] | None, content: str | None,
) -> int | None:
    _, user_number = normalize_user_id(user_id)
    run = _lock_run(db, user_number=user_number, conversation_id=conversation_id, run_id=run_id)
    response_status = str(response.get("status") or "failed")
    metadata = _json_dict(run.metadata_json)
    metadata.update({
        "latency_ms": latency_ms, "artifact_refs": response.get("artifact_refs") or [],
        "error": response.get("error"), **(extra_meta or {}), "terminal_event_pending": True,
    })
    if response_status == "confirmation_required":
        pending = response.get("pending_action")
        interrupt = response.get("interrupt")
        if not isinstance(pending, dict):
            raise PendingActionConflictError("The run has no pending action")
        run.status = "WAITING_INPUT"
        run.pending_interrupt_id = _interrupt_id(interrupt, pending)
        run.pending_interrupt_payload = _serialize({"interrupt": interrupt, "pending_action": pending})
        run.metadata_json = _serialize(metadata)
        run.modifier_id = user_number
        run.modification_time = _utc_datetime()
        return None
    run_status = _run_status(response_status)
    if run_status in {"FAILED", "CANCELLED"}:
        _fail_incomplete_tool_results(db, run_id=run_id, user_number=user_number)
    reply = content or str(response.get("reply") or "") or str(response.get("error") or "")
    if not reply:
        reply = "本次任务未产生可展示的回复。"
    message = _add_message(
        db, user_number=user_number, conversation_id=conversation_id, run_id=run_id,
        role="assistant", blocks=[{"type": "text", "text": reply}],
        status="COMPLETED" if run_status == "SUCCEEDED" else "FAILED",
        payload={"kind": "terminal", **metadata},
    )
    db.flush()
    run.status = run_status
    run.output_message_id = int(message.id)
    run.pending_interrupt_id = None
    run.pending_interrupt_payload = None
    run.error_code = _error_code(response)
    run.error_message = _error_message(response)
    run.metadata_json = _serialize(metadata)
    run.modifier_id = user_number
    run.modification_time = _utc_datetime()
    return int(message.id)


def claim_pending_action(
    db: Session, *, user_id: str, conversation_id: str, run_id: str, action_id: str,
    decision: str, comment: str | None,
) -> bool:
    _, user_number = normalize_user_id(user_id)
    run = _lock_run(db, user_number=user_number, conversation_id=conversation_id, run_id=run_id)
    metadata = _json_dict(run.metadata_json)
    approvals = metadata.get("approval_history")
    if (
        run.status == "RUNNING"
        and isinstance(approvals, list)
        and any(isinstance(item, dict) and item.get("action_id") == action_id for item in approvals)
    ):
        return False
    pending_action = _pending_payload(run).get("pending_action")
    if not isinstance(pending_action, dict) or str(pending_action.get("action_id") or "") != action_id:
        raise PendingActionConflictError("The requested action does not match the pending action")
    if run.status == "RUNNING":
        return False
    if run.status != "WAITING_INPUT":
        raise PendingActionConflictError("The run is not waiting for confirmation")
    approvals = list(approvals or [])
    approvals.append({"action_id": action_id, "decision": decision, "comment": comment or None,
                      "resolved_at": utc_now()})
    metadata["approval_history"] = approvals
    metadata.pop("terminal_event_sequence", None)
    metadata.pop("terminal_event_pending", None)
    run.status = "RUNNING"
    run.pending_interrupt_id = None
    run.pending_interrupt_payload = None
    run.metadata_json = _serialize(metadata)
    run.modifier_id = user_number
    run.modification_time = _utc_datetime()
    return True


def record_terminal_event_sequence(
    db: Session, *, user_id: str, run_id: str, sequence: int
) -> None:
    _, user_number = normalize_user_id(user_id)
    run = db.scalar(select(AgentRun).where(
        AgentRun.run_id == validate_run_id(run_id), AgentRun.creator_id == user_number,
    ).with_for_update())
    if run is None:
        raise ConversationNotFoundError("run not found")
    metadata = _json_dict(run.metadata_json)
    metadata["terminal_event_sequence"] = sequence
    metadata.pop("terminal_event_pending", None)
    run.metadata_json = _serialize(metadata)
    run.modifier_id = user_number
    run.modification_time = _utc_datetime()


def mark_interrupted(
    db: Session, *, user_id: str, conversation_id: str, run_id: str
) -> None:
    _, user_number = normalize_user_id(user_id)
    run = _lock_run(db, user_number=user_number, conversation_id=conversation_id, run_id=run_id)
    if run.status in TERMINAL_RUN_STATUSES:
        return
    _fail_incomplete_tool_results(db, run_id=run_id, user_number=user_number)
    message = _add_message(
        db, user_number=user_number, conversation_id=conversation_id, run_id=run_id,
        role="assistant", blocks=[{"type": "text", "text": "本次回复未完成。"}],
        status="FAILED", payload={"kind": "terminal", "interrupted": True},
    )
    db.flush()
    run.status = "FAILED"
    run.output_message_id = int(message.id)
    run.error_code = "INTERRUPTED"
    run.error_message = "本次回复未完成。"
    run.modifier_id = user_number
    run.modification_time = _utc_datetime()


def list_conversations(db: Session, *, user_id: str, limit: int) -> list[dict[str, Any]]:
    _, user_number = normalize_user_id(user_id)
    conversations = list(db.scalars(select(AgentConversation).where(
        AgentConversation.creator_id == user_number,
    ).order_by(AgentConversation.modification_time.desc(), AgentConversation.id.desc()).limit(limit)))
    items: list[dict[str, Any]] = []
    for conversation in conversations:
        messages = list(db.scalars(select(AgentMessage).where(
            AgentMessage.creator_id == user_number,
            AgentMessage.conversation_id == conversation.conversation_id,
        ).order_by(AgentMessage.id.desc())))
        latest = messages[0] if messages else None
        first_input = next((message for message in reversed(messages)
                            if message.role == "user" and _payload_kind(message.payload) == "input"), None)
        title = str(conversation.title or "").strip()
        if not title and first_input is not None:
            title = _display_text(decode_content(first_input.content))[:60]
        items.append({
            "conversation_id": conversation.conversation_id,
            "title": title or "(空会话)",
            "last_message_preview": (
                f"{'你' if latest.role == 'user' else '助手'}：{_display_text(decode_content(latest.content))[:80]}"
                if latest is not None else ""
            ),
            "message_count": len(messages),
            "last_message_at": _datetime_text(
                latest.creation_time if latest is not None else conversation.modification_time
            ),
        })
    return items


def list_messages(
    db: Session, *, user_id: str, conversation_id: str, limit: int, before_id: int | None
) -> tuple[list[dict[str, Any]], int | None]:
    _, user_number = normalize_user_id(user_id)
    conversation_id = validate_conversation_id(conversation_id)
    if _conversation_for_user(db, user_number=user_number, conversation_id=conversation_id) is None:
        raise ConversationNotFoundError("conversation not found")
    stmt: Select[tuple[AgentMessage]] = select(AgentMessage).where(
        AgentMessage.creator_id == user_number, AgentMessage.conversation_id == conversation_id,
    )
    if before_id is not None:
        stmt = stmt.where(AgentMessage.id < before_id)
    rows = list(db.scalars(stmt.order_by(AgentMessage.id.desc()).limit(limit + 1)))
    has_more = len(rows) > limit
    rows = rows[:limit]
    rows.reverse()
    items = [_message_view(row) for row in rows]
    return items, items[0]["id"] if has_more and items else None


def get_run_scope(db: Session, run_id: str) -> RunScope | None:
    run = db.scalar(select(AgentRun).where(AgentRun.run_id == validate_run_id(run_id)))
    return RunScope(user_id=str(run.creator_id), conversation_id=run.conversation_id) if run else None


def get_run_event_state(db: Session, *, user_id: str, run_id: str) -> RunEventState | None:
    _, user_number = normalize_user_id(user_id)
    run = db.scalar(select(AgentRun).where(
        AgentRun.run_id == validate_run_id(run_id), AgentRun.creator_id == user_number,
    ))
    if run is None:
        return None
    return RunEventState(
        status=_public_run_status(run.status),
        terminal_event_sequence=_positive_int(_json_dict(run.metadata_json).get("terminal_event_sequence")),
    )


def get_run_view(
    db: Session, *, user_id: str, conversation_id: str, run_id: str
) -> RunView:
    _, user_number = normalize_user_id(user_id)
    run = db.scalar(select(AgentRun).where(
        AgentRun.run_id == validate_run_id(run_id), AgentRun.creator_id == user_number,
        AgentRun.conversation_id == validate_conversation_id(conversation_id),
    ))
    if run is None:
        raise ConversationNotFoundError("run not found")
    output = db.get(AgentMessage, run.output_message_id) if run.output_message_id else None
    metadata = _json_dict(run.metadata_json)
    reply = _display_text(decode_content(output.content)) if output is not None else ""
    return RunView(
        run_id=run.run_id, conversation_id=run.conversation_id,
        status=_public_run_status(run.status), reply=reply,
        result={"status": _public_run_status(run.status), "reply": reply,
                "error": run.error_message, "error_code": run.error_code,
                "pending_action": _pending_payload(run).get("pending_action"),
                "artifact_refs": metadata.get("artifact_refs") or []},
        terminal_event_sequence=_positive_int(metadata.get("terminal_event_sequence")),
    )


def list_completed_turns_after(
    db: Session, *, user_id: str, conversation_id: str,
    after_assistant_message_id: int | None, limit: int,
) -> list[HistoryTurn]:
    if limit < 1:
        raise ValueError("limit must be positive")
    _, user_number = normalize_user_id(user_id)
    runs = list(db.scalars(select(AgentRun).where(
        AgentRun.creator_id == user_number,
        AgentRun.conversation_id == validate_conversation_id(conversation_id),
        AgentRun.status == "SUCCEEDED", AgentRun.input_message_id.is_not(None),
        AgentRun.output_message_id.is_not(None),
    ).order_by(AgentRun.output_message_id.asc())))
    turns: list[HistoryTurn] = []
    for run in runs:
        if after_assistant_message_id is not None and run.output_message_id <= after_assistant_message_id:
            continue
        user_message = db.get(AgentMessage, run.input_message_id)
        assistant_message = db.get(AgentMessage, run.output_message_id)
        if user_message is None or assistant_message is None:
            continue
        completed_at = assistant_message.creation_time
        if completed_at.tzinfo is None:
            completed_at = completed_at.replace(tzinfo=UTC)
        turns.append(HistoryTurn(
            user_message_id=int(user_message.id), assistant_message_id=int(assistant_message.id),
            user_content=_display_text(decode_content(user_message.content)),
            assistant_content=_display_text(decode_content(assistant_message.content)),
            completed_at=completed_at,
        ))
        if len(turns) == limit:
            break
    return turns


def delete_conversation(db: Session, *, user_id: str, conversation_id: str) -> None:
    _, user_number = normalize_user_id(user_id)
    conversation = _locked_conversation(
        db, user_number=user_number, conversation_id=validate_conversation_id(conversation_id)
    )
    db.execute(delete(AgentMessage).where(
        AgentMessage.creator_id == user_number, AgentMessage.conversation_id == conversation.conversation_id,
    ))
    db.execute(delete(AgentRun).where(
        AgentRun.creator_id == user_number, AgentRun.conversation_id == conversation.conversation_id,
    ))
    db.delete(conversation)


def begin_conversation_deletion(
    db: Session, *, user_id: str, conversation_id: str
) -> str:
    """Block new runs before the cross-store deletion sequence begins."""
    _, user_number = normalize_user_id(user_id)
    conversation = _locked_conversation(
        db, user_number=user_number, conversation_id=validate_conversation_id(conversation_id)
    )
    thread_id = _thread_id(conversation)
    if not thread_id:
        raise ConversationThreadError("conversation thread_id is required")
    conversation.status = "DELETED"
    conversation.modifier_id = user_number
    conversation.modification_time = _utc_datetime()
    return thread_id


def _conversation_for_user(db: Session, *, user_number: int, conversation_id: str) -> AgentConversation | None:
    return db.scalar(select(AgentConversation).where(
        AgentConversation.creator_id == user_number, AgentConversation.conversation_id == conversation_id,
    ))


def _locked_conversation(db: Session, *, user_number: int, conversation_id: str) -> AgentConversation:
    conversation = db.scalar(select(AgentConversation).where(
        AgentConversation.creator_id == user_number, AgentConversation.conversation_id == conversation_id,
    ).with_for_update())
    if conversation is None:
        raise ConversationNotFoundError("conversation not found")
    return conversation


def _require_runnable_conversation(conversation: AgentConversation) -> None:
    if conversation.status != "ACTIVE":
        raise ConversationStateError("conversation is not active")
    if not _thread_id(conversation):
        raise ConversationThreadError("conversation thread_id is required")


def _thread_id(conversation: AgentConversation) -> str:
    return str(conversation.thread_id or "").strip()


def _lock_run(db: Session, *, user_number: int, conversation_id: str, run_id: str) -> AgentRun:
    run = db.scalar(select(AgentRun).where(
        AgentRun.creator_id == user_number,
        AgentRun.conversation_id == validate_conversation_id(conversation_id),
        AgentRun.run_id == validate_run_id(run_id),
    ).with_for_update())
    if run is None:
        raise ConversationNotFoundError("run not found")
    return run


def _find_batch_message(
    db: Session, *, run_id: str, role: str, batch_id: str, kind: str
) -> AgentMessage | None:
    rows = list(db.scalars(select(AgentMessage).where(
        AgentMessage.run_id == run_id, AgentMessage.role == role,
    ).order_by(AgentMessage.id.asc()).with_for_update()))
    for row in rows:
        payload = _json_dict(row.payload)
        if payload.get("kind") == kind and payload.get("batch_id") == batch_id:
            return row
    return None


def _add_message(
    db: Session, *, user_number: int, conversation_id: str, run_id: str | None,
    role: str, blocks: list[dict[str, Any]], status: str, payload: dict[str, Any] | None,
) -> AgentMessage:
    message = AgentMessage(
        creator_id=user_number, modifier_id=user_number, conversation_id=conversation_id,
        run_id=run_id, role=role, content=_serialize(blocks),
        payload=_serialize(payload) if payload else None, status=status,
        creation_time=_utc_datetime(), modification_time=_utc_datetime(),
    )
    db.add(message)
    return message


def _tool_call_blocks(batch_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"type": "tool_use", "id": str(call["id"]), "name": str(call["name"]),
             "input": call.get("args") or call.get("input") or {}} for call in batch_calls]


def _tool_call_manifest(batch_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "id": str(call["id"]),
        "name": str(call["name"]),
        "kind": str(call.get("kind") or "tool"),
        "requires_confirmation": bool(call.get("requires_confirmation")),
    } for call in batch_calls]


def _tool_result_block(result: dict[str, Any]) -> dict[str, Any]:
    result_data = result.get("data")
    if result.get("action_type") == "subagent":
        result_content: Any = {
            **(result_data if isinstance(result_data, dict) else {"data": result_data}),
            "summary": result.get("summary"), "error_code": result.get("error_code"),
            "error_message": result.get("error_message"),
        }
    elif isinstance(result_data, dict) and set(result_data) == {"content"}:
        result_content = result_data["content"]
    else:
        result_content = result_data if result_data is not None else result.get("summary", "")
    return {"type": "tool_result", "tool_use_id": str(result["action_id"]),
            "content": result_content,
            "is_error": str(result.get("status") or "") not in {"success", "partial"}}


def _fail_incomplete_tool_results(
    db: Session, *, run_id: str, user_number: int
) -> None:
    messages = db.scalars(select(AgentMessage).where(
        AgentMessage.creator_id == user_number,
        AgentMessage.run_id == run_id,
        AgentMessage.role == "user",
        AgentMessage.status == "STREAMING",
    ).with_for_update())
    for message in messages:
        if _payload_kind(message.payload) == "tool_results":
            message.status = "FAILED"
            message.modifier_id = user_number
            message.modification_time = _utc_datetime()


def _blocks_or_empty(content: str) -> list[dict[str, Any]]:
    decoded = decode_content(content)
    return decoded if isinstance(decoded, list) else []


def decode_content(content: str) -> str | list[dict[str, Any]]:
    try:
        decoded = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return content
    if isinstance(decoded, list) and all(isinstance(item, dict) for item in decoded):
        return decoded
    return content


def _message_view(row: AgentMessage) -> dict[str, Any]:
    payload = _json_dict(row.payload)
    return {"id": int(row.id), "conversation_id": row.conversation_id, "run_id": row.run_id or "",
            "role": row.role, "content": decode_content(row.content),
            "status": _public_message_status(row.status), "source": str(payload.get("source") or "api"),
            "meta": payload or None, "created_at": _datetime_text(row.creation_time)}


def _pending_payload(run: AgentRun) -> dict[str, Any]:
    return _json_dict(run.pending_interrupt_payload)


def _interrupt_id(interrupt: Any, pending_action: dict[str, Any]) -> str:
    if isinstance(interrupt, dict):
        for key in ("id", "interrupt_id"):
            if interrupt.get(key):
                return str(interrupt[key])[:128]
    return str(pending_action.get("action_id") or "")[:128]


def _run_status(response_status: str) -> str:
    if response_status == "completed":
        return "SUCCEEDED"
    if response_status == "interrupted":
        return "CANCELLED"
    return "FAILED"


def _public_run_status(status: str) -> str:
    return {"PENDING": "running", "RUNNING": "running", "WAITING_INPUT": "confirmation_required",
            "SUCCEEDED": "completed", "FAILED": "failed", "CANCELLED": "interrupted"}.get(status, "failed")


def _public_message_status(status: str) -> str:
    return {"STREAMING": "running", "COMPLETED": "completed", "FAILED": "failed"}.get(status, "failed")


def _error_code(response: dict[str, Any]) -> str | None:
    result = response.get("last_action_result")
    if isinstance(result, dict) and result.get("error_code"):
        return str(result["error_code"])[:64]
    return "RUN_FAILED" if response.get("error") else None


def _error_message(response: dict[str, Any]) -> str | None:
    if response.get("error"):
        return str(response["error"])
    result = response.get("last_action_result")
    return str(result["error_message"]) if isinstance(result, dict) and result.get("error_message") else None


def _payload_kind(payload: str | None) -> str | None:
    kind = _json_dict(payload).get("kind")
    return str(kind) if kind is not None else None


def _display_text(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    return "".join(str(block.get("text") or "") for block in content if block.get("type") == "text")


def _serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _datetime_text(value: datetime) -> str:
    return value.replace(tzinfo=UTC).isoformat() if value.tzinfo is None else value.isoformat()


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _utc_datetime() -> datetime:
    """MySQL DATETIME values are stored as naive UTC timestamps."""
    return datetime.now(UTC).replace(tzinfo=None)
