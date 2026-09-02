from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.memory.schemas import HistoryTurn


TERMINAL_STATUSES = {
    "completed",
    "confirmation_required",
    "failed",
    "blocked",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _merge_card_items(
    previous: Any,
    current: Any,
    *,
    identity_key: str,
) -> list[dict[str, Any]]:
    if not isinstance(previous, list):
        previous = []
    if not isinstance(current, list):
        current = []

    merged = [dict(item) for item in previous if isinstance(item, dict)]
    positions = {
        str(item.get(identity_key)): index
        for index, item in enumerate(merged)
        if item.get(identity_key)
    }
    for item in current:
        if not isinstance(item, dict):
            continue
        item = dict(item)
        identity = str(item.get(identity_key) or "")
        if not identity or identity not in positions:
            positions[identity] = len(merged)
            merged.append(item)
            continue

        previous_item = merged[positions[identity]]
        if identity_key == "delegation_id":
            item["timeline"] = _merge_card_items(
                previous_item.get("timeline"),
                item.get("timeline"),
                identity_key="step_id",
            )
        merged[positions[identity]] = {**previous_item, **item}
    return merged


def _merge_cards(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    merged = {**previous, **current}
    merged["reasoning"] = _merge_card_items(
        previous.get("reasoning"),
        current.get("reasoning"),
        identity_key="reasoning_id",
    )
    merged["tools"] = _merge_card_items(
        previous.get("tools"),
        current.get("tools"),
        identity_key="action_id",
    )
    merged["subagents"] = _merge_card_items(
        previous.get("subagents"),
        current.get("subagents"),
        identity_key="delegation_id",
    )
    return merged


class PendingActionConflictError(ValueError):
    """The requested approval cannot be applied to the stored assistant card."""


class PendingActionClaim:
    def __init__(
        self,
        *,
        message_id: int,
        claimed: bool,
    ) -> None:
        self.message_id = message_id
        self.claimed = claimed


class ChatHistoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT '0',
                    conversation_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    role TEXT NOT NULL
                        CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL
                        CHECK (
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

                CREATE INDEX IF NOT EXISTS
                    ix_chat_log_run_id
                ON chat_log (run_id);
                """
            )
            self._migrate(connection)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
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

    async def start_turn(
        self,
        *,
        user_id: str = "0",
        conversation_id: str,
        run_id: str,
        user_content: str,
        source: str = "api",
    ) -> int:
        return await asyncio.to_thread(
            self._start_turn,
            user_id,
            conversation_id,
            run_id,
            user_content,
            source,
        )

    async def claim_pending_action(
        self,
        *,
        user_id: str = "0",
        conversation_id: str,
        run_id: str,
        action_id: str,
        decision: str,
        comment: str | None,
    ) -> PendingActionClaim:
        return await asyncio.to_thread(
            self._claim_pending_action,
            user_id,
            conversation_id,
            run_id,
            action_id,
            decision,
            comment,
        )

    async def complete_assistant_message(
        self,
        *,
        message_id: int,
        response: dict[str, Any],
        latency_ms: int,
        extra_meta: dict[str, Any] | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._complete_assistant_message,
            message_id,
            response,
            latency_ms,
            extra_meta,
        )

    async def mark_interrupted(
        self,
        *,
        message_id: int,
    ) -> None:
        await asyncio.to_thread(
            self._mark_interrupted,
            message_id,
        )

    async def list_conversations(
        self,
        *,
        user_id: str = "0",
        limit: int,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._list_conversations,
            user_id,
            limit,
        )

    async def list_messages(
        self,
        *,
        user_id: str = "0",
        conversation_id: str,
        limit: int,
        before_id: int | None,
    ) -> tuple[list[dict[str, Any]], int | None]:
        return await asyncio.to_thread(
            self._list_messages,
            user_id,
            conversation_id,
            limit,
            before_id,
        )

    async def list_completed_turns_after(
        self,
        *,
        user_id: str,
        conversation_id: str,
        after_assistant_message_id: int | None,
        limit: int,
    ) -> list[HistoryTurn]:
        return await asyncio.to_thread(
            self._list_completed_turns_after,
            user_id,
            conversation_id,
            after_assistant_message_id,
            limit,
        )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.path,
            timeout=5,
            isolation_level="IMMEDIATE",
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _start_turn(
        self,
        user_id: str,
        conversation_id: str,
        run_id: str,
        user_content: str,
        source: str,
    ) -> int:
        created_at = _utc_now()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_log (
                    user_id, conversation_id, run_id, role, content,
                    status, source, created_at
                )
                VALUES (?, ?, ?, 'user', ?, 'completed', ?, ?)
                """,
                (
                    user_id,
                    conversation_id,
                    run_id,
                    user_content,
                    source,
                    created_at,
                ),
            )

            cursor = connection.execute(
                """
                INSERT INTO chat_log (
                    user_id, conversation_id, run_id, role, content,
                    status, source, created_at
                )
                VALUES (?, ?, ?, 'assistant', '', 'running', ?, ?)
                """,
                (
                    user_id,
                    conversation_id,
                    run_id,
                    source,
                    created_at,
                ),
            )

            return int(cursor.lastrowid)

    def _claim_pending_action(
        self,
        user_id: str,
        conversation_id: str,
        run_id: str,
        action_id: str,
        decision: str,
        comment: str | None,
    ) -> PendingActionClaim:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, status, meta
                FROM chat_log
                WHERE conversation_id = ?
                  AND user_id = ?
                  AND run_id = ?
                  AND role = 'assistant'
                ORDER BY id DESC
                LIMIT 1
                """,
                (
                    conversation_id,
                    user_id,
                    run_id,
                ),
            ).fetchone()
            if row is None:
                raise PendingActionConflictError(
                    "No assistant confirmation exists for this run"
                )

            meta = json.loads(row["meta"]) if row["meta"] else {}
            card = meta.get("card")
            card = dict(card) if isinstance(card, dict) else {}
            approval_history = card.get("approval_history")
            approval_history = (
                list(approval_history)
                if isinstance(approval_history, list)
                else []
            )
            existing_decision = next(
                (
                    item.get("decision")
                    for item in approval_history
                    if isinstance(item, dict)
                    and item.get("action_id") == action_id
                ),
                None,
            )
            if row["status"] == "running" and existing_decision == decision:
                return PendingActionClaim(
                    message_id=int(row["id"]),
                    claimed=False,
                )
            if row["status"] == "running":
                raise PendingActionConflictError(
                    "The pending action is already being processed"
                )
            if row["status"] != "confirmation_required":
                raise PendingActionConflictError(
                    "The pending action has already been resolved"
                )

            pending_action = card.get("pending_action") or meta.get(
                "pending_action"
            )
            if not isinstance(pending_action, dict):
                raise PendingActionConflictError(
                    "The assistant message has no pending action"
                )
            if str(pending_action.get("action_id") or "") != action_id:
                raise PendingActionConflictError(
                    "The requested action does not match the pending action"
                )

            approval_history.append(
                {
                    "action_id": action_id,
                    "action_type": pending_action.get("action_type"),
                    "name": pending_action.get("name"),
                    "decision": decision,
                    "comment": comment or None,
                    "resolved_at": _utc_now(),
                }
            )
            card.update(
                {
                    "status": "running",
                    "pending_action": None,
                    "approval_history": approval_history,
                    "error": None,
                }
            )
            meta["card"] = card
            meta.pop("pending_action", None)

            cursor = connection.execute(
                """
                UPDATE chat_log
                SET status = 'running', meta = ?
                WHERE id = ?
                  AND role = 'assistant'
                  AND status = 'confirmation_required'
                """,
                (
                    json.dumps(meta, ensure_ascii=False, default=str),
                    row["id"],
                ),
            )
            if cursor.rowcount != 1:
                raise PendingActionConflictError(
                    "The pending action was resolved concurrently"
                )
            return PendingActionClaim(
                message_id=int(row["id"]),
                claimed=True,
            )

    def _complete_assistant_message(
        self,
        message_id: int,
        response: dict[str, Any],
        latency_ms: int,
        extra_meta: dict[str, Any] | None,
    ) -> None:
        status = str(response.get("status") or "failed")
        if status not in TERMINAL_STATUSES:
            status = "failed"

        content = (
            str(response.get("reply") or "")
            or str(response.get("error") or "")
            or "本次任务未产生可展示的回复。"
        )

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT meta FROM chat_log WHERE id = ? AND role = 'assistant'",
                (message_id,),
            ).fetchone()
            existing_meta = (
                json.loads(existing["meta"])
                if existing is not None and existing["meta"]
                else {}
            )
            existing_card = existing_meta.get("card")
            existing_card = (
                dict(existing_card)
                if isinstance(existing_card, dict)
                else {}
            )
            incoming_card = (extra_meta or {}).get("card")
            incoming_card = (
                dict(incoming_card) if isinstance(incoming_card, dict) else {}
            )
            extra_card = _merge_cards(existing_card, incoming_card)
            approval_history = existing_card.get("approval_history")
            if isinstance(approval_history, list):
                extra_card["approval_history"] = approval_history
            if extra_card:
                extra_card["status"] = status
                extra_card["pending_action"] = response.get("pending_action")
                extra_meta = {
                    **(extra_meta or {}),
                    "card": extra_card,
                }

            meta = {
                "latency_ms": latency_ms,
                "artifact_refs": response.get("artifact_refs") or [],
                "last_action_result": response.get("last_action_result"),
                "pending_action": response.get("pending_action"),
                "error": response.get("error"),
                **(extra_meta or {}),
            }
            meta = {
                key: value
                for key, value in meta.items()
                if value not in (None, [], "")
            }

            cursor = connection.execute(
                """
                UPDATE chat_log
                SET content = ?, status = ?, meta = ?
                WHERE id = ? AND role = 'assistant' AND status = 'running'
                """,
                (
                    content,
                    status,
                    json.dumps(meta, ensure_ascii=False, default=str)
                    if meta
                    else None,
                    message_id,
                ),
            )

            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"Assistant chat log {message_id} is not running"
                )

    def _mark_interrupted(
        self,
        message_id: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE chat_log
                SET status = 'interrupted',
                    content = '本次回复未完成。'
                WHERE id = ? AND role = 'assistant' AND status = 'running'
                """,
                (message_id,),
            )

    def _list_conversations(
        self,
        user_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        query = """
            WITH conversations AS (
                SELECT
                    user_id,
                    conversation_id,
                    COUNT(*) AS message_count,
                    MAX(id) AS latest_message_id
                FROM chat_log
                WHERE user_id = ?
                GROUP BY user_id, conversation_id
            )
            SELECT
                c.conversation_id,
                c.message_count,
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
        """

        with self._connect() as connection:
            rows = connection.execute(query, (user_id, limit)).fetchall()

        return [
            {
                "conversation_id": row["conversation_id"],
                "title": row["title"][:60],
                "last_message_preview": (
                    f"{'你' if row['last_message_role'] == 'user' else '助手'}："
                    f"{row['last_message_content'][:80]}"
                ),
                "message_count": row["message_count"],
                "last_message_at": row["last_message_at"],
            }
            for row in rows
        ]

    def _list_messages(
        self,
        user_id: str,
        conversation_id: str,
        limit: int,
        before_id: int | None,
    ) -> tuple[list[dict[str, Any]], int | None]:
        sql = """
        SELECT
            id,
            conversation_id,
            run_id,
            role,
            content,
            status,
            source,
            meta,
            created_at
        FROM chat_log
        WHERE user_id = ? AND conversation_id = ?
        """

        params: list[Any] = [user_id, conversation_id]

        if before_id is not None:
            sql += " AND id < ?"
            params.append(before_id)

        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit + 1)

        with self._connect() as connection:
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
                "content": row["content"],
                "status": row["status"],
                "source": row["source"],
                "meta": json.loads(row["meta"]) if row["meta"] else None,
                "created_at": row["created_at"],
            }
            for row in rows
        ]

        next_before_id = items[0]["id"] if has_more and items else None

        return items, next_before_id

    def _list_completed_turns_after(
        self,
        user_id: str,
        conversation_id: str,
        after_assistant_message_id: int | None,
        limit: int,
    ) -> list[HistoryTurn]:
        if limit < 1:
            raise ValueError("limit must be positive")

        sql = """
            SELECT
                user_message.id AS user_message_id,
                assistant_message.id AS assistant_message_id,
                user_message.content AS user_content,
                assistant_message.content AS assistant_content,
                assistant_message.created_at AS completed_at
            FROM chat_log user_message
            JOIN chat_log assistant_message
                ON assistant_message.run_id = user_message.run_id
                AND assistant_message.user_id = user_message.user_id
                AND assistant_message.role = 'assistant'
            WHERE user_message.role = 'user'
                AND user_message.user_id = ?
                AND user_message.conversation_id = ?
                AND assistant_message.conversation_id = ?
                AND assistant_message.status = 'completed'
        """
        params: list[Any] = [user_id, conversation_id, conversation_id]
        if after_assistant_message_id is not None:
            sql += " AND assistant_message.id > ?"
            params.append(after_assistant_message_id)
        sql += " ORDER BY assistant_message.id ASC LIMIT ?"
        params.append(limit)

        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()

        return [
            HistoryTurn(
                user_message_id=int(row["user_message_id"]),
                assistant_message_id=int(row["assistant_message_id"]),
                user_content=row["user_content"],
                assistant_content=row["assistant_content"],
                completed_at=datetime.fromisoformat(row["completed_at"]),
            )
            for row in rows
        ]


_history_store: ChatHistoryStore | None = None


def initialize_history_store(
    db_path: str | Path,
) -> ChatHistoryStore:
    global _history_store

    _history_store = ChatHistoryStore(db_path)
    _history_store.initialize()

    return _history_store


def get_history_store() -> ChatHistoryStore:
    if _history_store is None:
        raise RuntimeError("Chat history store has not been initialized")

    return _history_store
