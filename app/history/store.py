from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = {
    "completed",
    "confirmation_required",
    "failed",
    "blocked",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


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
                    ix_chat_log_conversation_id_id
                ON chat_log (conversation_id, id);

                CREATE INDEX IF NOT EXISTS
                    ix_chat_log_run_id
                ON chat_log (run_id);
                """
            )

    async def start_turn(
        self,
        *,
        conversation_id: str,
        run_id: str,
        user_content: str,
        source: str = "api",
    ) -> int:
        return await asyncio.to_thread(
            self._start_turn,
            conversation_id,
            run_id,
            user_content,
            source,
        )

    async def start_resume(
        self,
        *,
        conversation_id: str,
        run_id: str,
        source: str = "api",
    ) -> int:
        return await asyncio.to_thread(
            self._start_assistant_message,
            conversation_id,
            run_id,
            source,
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
        limit: int,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._list_conversations,
            limit,
        )

    async def list_messages(
        self,
        *,
        conversation_id: str,
        limit: int,
        before_id: int | None,
    ) -> tuple[list[dict[str, Any]], int | None]:
        return await asyncio.to_thread(
            self._list_messages,
            conversation_id,
            limit,
            before_id,
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
                    conversation_id, run_id, role, content,
                    status, source, created_at
                )
                VALUES (?, ?, 'user', ?, 'completed', ?, ?)
                """,
                (
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
                    conversation_id, run_id, role, content,
                    status, source, created_at
                )
                VALUES (?, ?, 'assistant', '', 'running', ?, ?)
                """,
                (
                    conversation_id,
                    run_id,
                    source,
                    created_at,
                ),
            )

            return int(cursor.lastrowid)

    def _start_assistant_message(
        self,
        conversation_id: str,
        run_id: str,
        source: str,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO chat_log (
                    conversation_id, run_id, role, content,
                    status, source, created_at
                )
                VALUES (?, ?, 'assistant', '', 'running', ?, ?)
                """,
                (
                    conversation_id,
                    run_id,
                    source,
                    _utc_now(),
                ),
            )
            return int(cursor.lastrowid)

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

        meta = {
            "latency_ms": latency_ms,
            "artifact_refs": response.get("artifact_refs") or [],
            "last_action_result": response.get("last_action_result"),
            "pending_action": response.get("pending_action"),
            "error": response.get("error"),
        }
        if extra_meta:
            meta.update(extra_meta)
        meta = {
            key: value
            for key, value in meta.items()
            if value not in (None, [], "")
        }

        with self._connect() as connection:
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
        limit: int,
    ) -> list[dict[str, Any]]:
        query = """
            WITH conversations AS (
                SELECT
                    conversation_id,
                    COUNT(*) AS message_count,
                    MAX(id) AS latest_message_id
                FROM chat_log
                GROUP BY conversation_id
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
            rows = connection.execute(query, (limit,)).fetchall()

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
        WHERE conversation_id = ?
        """

        params: list[Any] = [conversation_id]

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
