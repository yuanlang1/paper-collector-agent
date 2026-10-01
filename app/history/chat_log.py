from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.history.sqlite import HistoryDatabase
from app.memory.schemas import HistoryTurn


TERMINAL_STATUSES = {
    "completed",
    "confirmation_required",
    "failed",
    "blocked",
}

_CANONICAL_RUN_ID = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,118}[A-Za-z0-9])?"
)


class PendingActionConflictError(ValueError):
    """The requested approval cannot be applied to the stored tool call."""


class InvalidRunIdError(ValueError):
    """The run ID cannot be used as a stable artifact directory name."""


class RunAlreadyExistsError(ValueError):
    """The requested root run has already been created."""


@dataclass(frozen=True)
class PendingActionClaim:
    message_id: int
    claimed: bool


@dataclass(frozen=True)
class RunScope:
    user_id: str
    conversation_id: str


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS chat_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL DEFAULT '0',
            conversation_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL CHECK (
                status IN (
                    'running',
                    'completed',
                    'confirmation_required',
                    'failed',
                    'blocked',
                    'interrupted'
                )
            ),
            source TEXT NOT NULL DEFAULT 'api',
            meta TEXT,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS ix_chat_log_run_id
        ON chat_log (run_id);
        """
    )
    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(chat_log)").fetchall()
    }
    if "user_id" not in columns:
        connection.execute(
            "ALTER TABLE chat_log ADD COLUMN user_id TEXT NOT NULL DEFAULT '0'"
        )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS ix_chat_log_user_conversation_id "
        "ON chat_log (user_id, conversation_id, id)"
    )


def validate_run_id(value: str) -> str:
    if not _CANONICAL_RUN_ID.fullmatch(value):
        raise InvalidRunIdError(
            "run_id must be 1-120 ASCII letters, digits, '.', '_' or '-', "
            "and must start and end with a letter or digit"
        )
    return value


def start_turn(
    database: HistoryDatabase,
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    user_content: str,
    source: str = "api",
) -> int:
    run_id = validate_run_id(run_id)
    created_at = utc_now()
    with database.transaction() as connection:
        existing = connection.execute(
            "SELECT 1 FROM chat_log WHERE run_id = ? LIMIT 1",
            (run_id,),
        ).fetchone()
        if existing is not None:
            raise RunAlreadyExistsError("run_id already exists")
        cursor = connection.execute(
            """
            INSERT INTO chat_log (
                user_id, conversation_id, run_id, role, content,
                status, source, created_at
            )
            VALUES (?, ?, ?, 'user', ?, 'completed', ?, ?)
            """,
            (user_id, conversation_id, run_id, user_content, source, created_at),
        )
        return int(cursor.lastrowid)


def append_message_in_transaction(
    connection: sqlite3.Connection,
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    role: str,
    content: str | list[dict[str, Any]],
    status: str,
    source: str = "api",
    meta: dict[str, Any] | None = None,
) -> int:
    if role not in {"user", "assistant"}:
        raise ValueError("role must be user or assistant")
    if status not in TERMINAL_STATUSES | {"running", "interrupted"}:
        raise ValueError(f"Unsupported chat status: {status}")
    serialized_content = (
        json.dumps(content, ensure_ascii=False, default=str)
        if isinstance(content, list)
        else content
    )
    clean_meta = {
        key: value
        for key, value in (meta or {}).items()
        if key != "card" and value not in (None, [], "")
    }
    cursor = connection.execute(
        """
        INSERT INTO chat_log (
            user_id, conversation_id, run_id, role, content,
            status, source, meta, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            conversation_id,
            run_id,
            role,
            serialized_content,
            status,
            source,
            json.dumps(clean_meta, ensure_ascii=False, default=str)
            if clean_meta
            else None,
            utc_now(),
        ),
    )
    return int(cursor.lastrowid)


def append_action_started_in_transaction(
    connection: sqlite3.Connection,
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    action: dict[str, Any],
) -> int:
    action_id = str(action["action_id"])
    return append_message_in_transaction(
        connection,
        user_id=user_id,
        conversation_id=conversation_id,
        run_id=run_id,
        role="assistant",
        content=[
            {
                "type": "tool_use",
                "id": action_id,
                "name": str(action["name"]),
                "input": action.get("input") or {},
            }
        ],
        status="running",
        meta={
            "action_id": action_id,
            "action_type": action.get("action_type") or "tool",
            "requires_confirmation": bool(action.get("requires_confirmation")),
        },
    )


def append_action_result_in_transaction(
    connection: sqlite3.Connection,
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    result: dict[str, Any],
) -> int:
    action_id = str(result["action_id"])
    result_data = result.get("data")
    if isinstance(result_data, dict) and set(result_data) == {"content"}:
        result_content: Any = result_data["content"]
    else:
        result_content = result_data if result_data is not None else result.get("summary", "")
    is_error = str(result.get("status") or "") in {"error", "rejected"}
    message_id = append_message_in_transaction(
        connection,
        user_id=user_id,
        conversation_id=conversation_id,
        run_id=run_id,
        role="user",
        content=[
            {
                "type": "tool_result",
                "tool_use_id": action_id,
                "content": result_content,
                "is_error": is_error,
            }
        ],
        status="completed",
    )
    connection.execute(
        """
        UPDATE chat_log
        SET status = 'completed'
        WHERE user_id = ?
          AND conversation_id = ?
          AND run_id = ?
          AND role = 'assistant'
          AND status = 'running'
          AND meta LIKE ?
        """,
        (user_id, conversation_id, run_id, f'%"action_id": "{action_id}"%'),
    )
    return message_id


def append_terminal_assistant_message_in_transaction(
    connection: sqlite3.Connection,
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    response: dict[str, Any],
    latency_ms: int,
    extra_meta: dict[str, Any] | None = None,
    content: str | None = None,
) -> int | None:
    status = str(response.get("status") or "failed")
    if status == "confirmation_required":
        mark_pending_action_in_transaction(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
            run_id=run_id,
            pending_action=response.get("pending_action"),
        )
        return None
    if status not in TERMINAL_STATUSES | {"interrupted"}:
        status = "failed"
    reply = content or str(response.get("reply") or "") or str(response.get("error") or "")
    if not reply:
        reply = "本次任务未产生可展示的回复。"
    meta = {
        "latency_ms": latency_ms,
        "artifact_refs": response.get("artifact_refs") or [],
        "error": response.get("error"),
        **(extra_meta or {}),
    }
    return append_message_in_transaction(
        connection,
        user_id=user_id,
        conversation_id=conversation_id,
        run_id=run_id,
        role="assistant",
        content=[{"type": "text", "text": reply}],
        status=status,
        meta=meta,
    )


def mark_pending_action_in_transaction(
    connection: sqlite3.Connection,
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    pending_action: Any,
) -> None:
    if not isinstance(pending_action, dict):
        raise PendingActionConflictError("The run has no pending action")
    action_id = str(pending_action.get("action_id") or "")
    if not action_id:
        raise PendingActionConflictError("The pending action has no action ID")
    row = connection.execute(
        """
        SELECT id, meta
        FROM chat_log
        WHERE user_id = ?
          AND conversation_id = ?
          AND run_id = ?
          AND role = 'assistant'
          AND status = 'running'
        ORDER BY id DESC
        """,
        (user_id, conversation_id, run_id),
    ).fetchone()
    if row is None:
        raise PendingActionConflictError("No running tool call exists for this run")
    meta = _parse_meta(row["meta"])
    if str(meta.get("action_id") or "") != action_id:
        raise PendingActionConflictError("The pending action does not match the tool call")
    meta["pending_action"] = pending_action
    cursor = connection.execute(
        """
        UPDATE chat_log
        SET status = 'confirmation_required', meta = ?
        WHERE id = ? AND status = 'running'
        """,
        (json.dumps(meta, ensure_ascii=False, default=str), row["id"]),
    )
    if cursor.rowcount != 1:
        raise PendingActionConflictError("The pending action was resolved concurrently")


def claim_pending_action(
    database: HistoryDatabase,
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    action_id: str,
    decision: str,
    comment: str | None,
) -> PendingActionClaim:
    with database.transaction() as connection:
        rows = connection.execute(
            """
            SELECT id, status, meta
            FROM chat_log
            WHERE conversation_id = ?
              AND user_id = ?
              AND run_id = ?
              AND role = 'assistant'
              AND status IN ('confirmation_required', 'running')
            ORDER BY id DESC
            """,
            (conversation_id, user_id, run_id),
        ).fetchall()
        if not rows:
            raise PendingActionConflictError("No assistant confirmation exists for this run")
        row = None
        meta: dict[str, Any] = {}
        pending_action: dict[str, Any] | None = None
        for candidate in rows:
            candidate_meta = _parse_meta(candidate["meta"])
            approval_history = candidate_meta.get("approval_history")
            if (
                candidate["status"] == "running"
                and isinstance(approval_history, list)
                and any(
                    isinstance(item, dict) and item.get("action_id") == action_id
                    for item in approval_history
                )
            ):
                return PendingActionClaim(message_id=int(candidate["id"]), claimed=False)
            candidate_pending = candidate_meta.get("pending_action")
            if (
                isinstance(candidate_pending, dict)
                and str(candidate_pending.get("action_id") or "") == action_id
            ):
                row = candidate
                meta = candidate_meta
                pending_action = candidate_pending
                break
        if row is None or pending_action is None:
            raise PendingActionConflictError("The requested action does not match the pending action")
        if row["status"] == "running":
            return PendingActionClaim(message_id=int(row["id"]), claimed=False)

        approval_history = meta.get("approval_history")
        approval_history = list(approval_history) if isinstance(approval_history, list) else []
        approval_history.append(
            {
                "action_id": action_id,
                "action_type": pending_action.get("action_type"),
                "name": pending_action.get("name"),
                "decision": decision,
                "comment": comment or None,
                "resolved_at": utc_now(),
            }
        )
        meta.pop("card", None)
        meta.pop("pending_action", None)
        meta["approval_history"] = approval_history
        cursor = connection.execute(
            """
            UPDATE chat_log
            SET status = 'running', meta = ?
            WHERE id = ? AND role = 'assistant' AND status = 'confirmation_required'
            """,
            (json.dumps(meta, ensure_ascii=False, default=str), row["id"]),
        )
        if cursor.rowcount != 1:
            raise PendingActionConflictError("The pending action was resolved concurrently")
        return PendingActionClaim(message_id=int(row["id"]), claimed=True)


def mark_interrupted(
    database: HistoryDatabase,
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    message_id: int | None = None,
) -> None:
    with database.transaction() as connection:
        append_terminal_assistant_message_in_transaction(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
            run_id=run_id,
            response={"status": "interrupted", "reply": "本次回复未完成。"},
            latency_ms=0,
        )


def list_conversations(
    database: HistoryDatabase,
    *,
    user_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    with database.transaction() as connection:
        rows = connection.execute(
            """
            WITH conversations AS (
                SELECT user_id, conversation_id, COUNT(*) AS message_count,
                       MAX(id) AS latest_message_id
                FROM chat_log
                WHERE user_id = ?
                GROUP BY user_id, conversation_id
            )
            SELECT c.conversation_id, c.message_count,
                   latest.content AS last_message_content,
                   latest.role AS last_message_role,
                   latest.created_at AS last_message_at,
                   COALESCE(
                       (
                           SELECT content
                           FROM chat_log first_user
                           WHERE first_user.conversation_id = c.conversation_id
                             AND first_user.user_id = c.user_id
                             AND first_user.role = 'user'
                           ORDER BY first_user.id ASC
                           LIMIT 1
                       ),
                       '(空会话)'
                   ) AS title
            FROM conversations c
            JOIN chat_log latest ON latest.id = c.latest_message_id
            ORDER BY c.latest_message_id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [
        {
            "conversation_id": row["conversation_id"],
            "title": _display_text(row["title"])[:60] or "(空会话)",
            "last_message_preview": (
                f"{'你' if row['last_message_role'] == 'user' else '助手'}："
                f"{_display_text(row['last_message_content'])[:80]}"
            ),
            "message_count": row["message_count"],
            "last_message_at": row["last_message_at"],
        }
        for row in rows
    ]


def list_messages(
    database: HistoryDatabase,
    *,
    user_id: str,
    conversation_id: str,
    limit: int,
    before_id: int | None,
) -> tuple[list[dict[str, Any]], int | None]:
    sql = """
        SELECT id, conversation_id, run_id, role, content, status, source, meta, created_at
        FROM chat_log
        WHERE user_id = ? AND conversation_id = ?
    """
    params: list[Any] = [user_id, conversation_id]
    if before_id is not None:
        sql += " AND id < ?"
        params.append(before_id)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit + 1)
    with database.transaction() as connection:
        rows = connection.execute(sql, params).fetchall()
    has_more = len(rows) > limit
    rows = rows[:limit]
    rows.reverse()
    items = [
        {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "run_id": row["run_id"],
            "role": row["role"],
            "content": decode_content(row["content"]),
            "status": row["status"],
            "source": row["source"],
            "meta": _parse_meta(row["meta"]) or None,
            "created_at": row["created_at"],
        }
        for row in rows
    ]
    return items, items[0]["id"] if has_more and items else None


def get_run_scope(database: HistoryDatabase, run_id: str) -> RunScope | None:
    with database.transaction() as connection:
        rows = connection.execute(
            """
            SELECT user_id, conversation_id
            FROM chat_log
            WHERE run_id = ?
            GROUP BY user_id, conversation_id
            LIMIT 2
            """,
            (run_id,),
        ).fetchall()
    if len(rows) != 1:
        return None
    return RunScope(user_id=str(rows[0]["user_id"]), conversation_id=str(rows[0]["conversation_id"]))


def list_run_ids(
    database: HistoryDatabase,
    *,
    user_id: str,
    conversation_id: str,
) -> tuple[str, ...]:
    with database.transaction() as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT run_id
            FROM chat_log
            WHERE user_id = ? AND conversation_id = ?
            ORDER BY run_id
            """,
            (user_id, conversation_id),
        ).fetchall()
    return tuple(str(row["run_id"]) for row in rows)


def get_run_status(
    database: HistoryDatabase,
    *,
    user_id: str,
    run_id: str,
) -> str | None:
    with database.transaction() as connection:
        row = connection.execute(
            """
            SELECT content, status
            FROM chat_log
            WHERE user_id = ? AND run_id = ? AND role = 'assistant'
            ORDER BY id DESC
            LIMIT 1
            """,
            (user_id, run_id),
        ).fetchone()
    if row is None:
        return None
    content = decode_content(row["content"])
    if row["status"] == "confirmation_required":
        return "confirmation_required"
    if _is_text_block(content) and row["status"] in {
        "completed",
        "failed",
        "blocked",
        "interrupted",
    }:
        return str(row["status"])
    return "running"


def list_completed_turns_after(
    database: HistoryDatabase,
    *,
    user_id: str,
    conversation_id: str,
    after_assistant_message_id: int | None,
    limit: int,
) -> list[HistoryTurn]:
    if limit < 1:
        raise ValueError("limit must be positive")
    with database.transaction() as connection:
        rows = connection.execute(
            """
            SELECT id, run_id, role, content, status, created_at
            FROM chat_log
            WHERE user_id = ? AND conversation_id = ?
            ORDER BY id ASC
            """,
            (user_id, conversation_id),
        ).fetchall()
    runs: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        runs.setdefault(str(row["run_id"]), []).append(row)
    turns: list[HistoryTurn] = []
    for run_rows in runs.values():
        user_row = next(
            (
                row
                for row in run_rows
                if row["role"] == "user" and _is_user_text(decode_content(row["content"]))
            ),
            None,
        )
        terminal_row = run_rows[-1]
        terminal_content = decode_content(terminal_row["content"])
        if (
            user_row is None
            or terminal_row["role"] != "assistant"
            or terminal_row["status"] != "completed"
            or not _is_text_block(terminal_content)
            or (
                after_assistant_message_id is not None
                and int(terminal_row["id"]) <= after_assistant_message_id
            )
        ):
            continue
        turns.append(
            HistoryTurn(
                user_message_id=int(user_row["id"]),
                assistant_message_id=int(terminal_row["id"]),
                user_content=_display_text(decode_content(user_row["content"])),
                assistant_content=_display_text(terminal_content),
                completed_at=datetime.fromisoformat(terminal_row["created_at"]),
            )
        )
    return sorted(turns, key=lambda turn: turn.assistant_message_id)[:limit]


def delete_conversation(
    connection: sqlite3.Connection,
    *,
    user_id: str,
    conversation_id: str,
) -> int:
    cursor = connection.execute(
        "DELETE FROM chat_log WHERE user_id = ? AND conversation_id = ?",
        (user_id, conversation_id),
    )
    return cursor.rowcount


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def decode_content(content: str) -> str | list[dict[str, Any]]:
    try:
        decoded = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return content
    if isinstance(decoded, list) and all(isinstance(item, dict) for item in decoded):
        return decoded
    return content


def _parse_meta(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _is_text_block(content: str | list[dict[str, Any]]) -> bool:
    return (
        isinstance(content, list)
        and len(content) == 1
        and content[0].get("type") == "text"
    )


def _is_user_text(content: str | list[dict[str, Any]]) -> bool:
    return isinstance(content, str) or _is_text_block(content)


def _display_text(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    text = [str(block.get("text") or "") for block in content if block.get("type") == "text"]
    if text:
        return "".join(text)
    tool = next((block for block in content if block.get("type") == "tool_use"), None)
    if isinstance(tool, dict):
        return f"调用工具：{tool.get('name') or ''}"
    result = next((block for block in content if block.get("type") == "tool_result"), None)
    if isinstance(result, dict):
        return str(result.get("content") or "")
    return ""
