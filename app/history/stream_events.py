from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from app.history.sqlite import HistoryDatabase


@dataclass(frozen=True)
class AgentStreamEvent:
    id: int
    user_id: str
    conversation_id: str
    run_id: str
    event_name: str
    data: dict[str, Any]


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS agent_stream_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            creator_id INTEGER NOT NULL DEFAULT 0,
            creation_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            modifier_id INTEGER NOT NULL DEFAULT 0,
            modification_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            conversation_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            event_name TEXT NOT NULL,
            data TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS ix_agent_stream_events_user_run_id
        ON agent_stream_events (user_id, run_id, id);

        CREATE INDEX IF NOT EXISTS ix_agent_stream_events_user_conversation_id
        ON agent_stream_events (user_id, conversation_id, id);
        """
    )


def append(
    database: HistoryDatabase,
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    event_name: str,
    data: dict[str, Any],
) -> AgentStreamEvent:
    with database.transaction() as connection:
        return insert(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
            run_id=run_id,
            event_name=event_name,
            data=data,
        )


def insert(
    connection: sqlite3.Connection,
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    event_name: str,
    data: dict[str, Any],
) -> AgentStreamEvent:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    cursor = connection.execute(
        """
        INSERT INTO agent_stream_events (
            user_id, conversation_id, run_id, event_name, data
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, conversation_id, run_id, event_name, payload),
    )
    return AgentStreamEvent(
        id=int(cursor.lastrowid),
        user_id=user_id,
        conversation_id=conversation_id,
        run_id=run_id,
        event_name=event_name,
        data=json.loads(payload),
    )


def get(
    database: HistoryDatabase,
    *,
    user_id: str,
    run_id: str,
    event_id: int,
) -> AgentStreamEvent | None:
    with database.transaction() as connection:
        row = connection.execute(
            """
            SELECT id, user_id, conversation_id, run_id, event_name, data
            FROM agent_stream_events
            WHERE id = ? AND user_id = ? AND run_id = ?
            """,
            (event_id, user_id, run_id),
        ).fetchone()
    return from_row(row) if row else None


def list_after(
    database: HistoryDatabase,
    *,
    user_id: str,
    run_id: str,
    after_id: int | None,
    limit: int,
) -> list[AgentStreamEvent]:
    if limit < 1:
        raise ValueError("limit must be positive")
    with database.transaction() as connection:
        rows = connection.execute(
            """
            SELECT id, user_id, conversation_id, run_id, event_name, data
            FROM agent_stream_events
            WHERE user_id = ?
              AND run_id = ?
              AND (? IS NULL OR id > ?)
            ORDER BY id
            LIMIT ?
            """,
            (user_id, run_id, after_id, after_id, limit),
        ).fetchall()
    return [from_row(row) for row in rows]


def delete_conversation(
    connection: sqlite3.Connection,
    *,
    user_id: str,
    conversation_id: str,
) -> None:
    connection.execute(
        """
        DELETE FROM agent_stream_events
        WHERE user_id = ? AND conversation_id = ?
        """,
        (user_id, conversation_id),
    )


def from_row(row: sqlite3.Row) -> AgentStreamEvent:
    data = json.loads(row["data"])
    if not isinstance(data, dict):
        raise ValueError("agent stream event data must be an object")
    return AgentStreamEvent(
        id=int(row["id"]),
        user_id=str(row["user_id"]),
        conversation_id=str(row["conversation_id"]),
        run_id=str(row["run_id"]),
        event_name=str(row["event_name"]),
        data=data,
    )
