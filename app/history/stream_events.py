from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from app.history.sqlite import HistoryDatabase


@dataclass(frozen=True)
class AgentStreamEvent:
    id: int
    sequence: int
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
            sequence INTEGER NOT NULL CHECK (sequence > 0),
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

        CREATE INDEX IF NOT EXISTS ix_agent_stream_events_user_conversation_id
        ON agent_stream_events (user_id, conversation_id, id);
        """
    )
    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(agent_stream_events)").fetchall()
    }
    if "sequence" not in columns:
        connection.execute("ALTER TABLE agent_stream_events ADD COLUMN sequence INTEGER")
    connection.execute(
        """
        WITH ranked AS (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY user_id, run_id
                       ORDER BY id
                   ) AS sequence
            FROM agent_stream_events
        )
        UPDATE agent_stream_events
        SET sequence = (
            SELECT ranked.sequence
            FROM ranked
            WHERE ranked.id = agent_stream_events.id
        )
        WHERE sequence IS NULL
        """
    )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_agent_stream_events_user_run_sequence "
        "ON agent_stream_events (user_id, run_id, sequence)"
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
    if not connection.in_transaction:
        connection.execute("BEGIN IMMEDIATE")
    payload = json.dumps(data, ensure_ascii=False, default=str)
    sequence = next_sequence(connection, user_id=user_id, run_id=run_id)
    cursor = connection.execute(
        """
        INSERT INTO agent_stream_events (
            sequence, user_id, conversation_id, run_id, event_name, data
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            sequence,
            user_id,
            conversation_id,
            run_id,
            event_name,
            payload,
        ),
    )
    return AgentStreamEvent(
        id=int(cursor.lastrowid),
        sequence=sequence,
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
    sequence: int,
) -> AgentStreamEvent | None:
    with database.transaction() as connection:
        row = connection.execute(
            """
            SELECT id, sequence, user_id, conversation_id, run_id, event_name, data
            FROM agent_stream_events
            WHERE sequence = ? AND user_id = ? AND run_id = ?
            """,
            (sequence, user_id, run_id),
        ).fetchone()
    return from_row(row) if row else None


def list_after(
    database: HistoryDatabase,
    *,
    user_id: str,
    run_id: str,
    after_sequence: int | None,
    limit: int,
) -> list[AgentStreamEvent]:
    if limit < 1:
        raise ValueError("limit must be positive")
    with database.transaction() as connection:
        rows = connection.execute(
            """
            SELECT id, sequence, user_id, conversation_id, run_id, event_name, data
            FROM agent_stream_events
            WHERE user_id = ?
              AND run_id = ?
              AND (? IS NULL OR sequence > ?)
            ORDER BY sequence
            LIMIT ?
            """,
            (user_id, run_id, after_sequence, after_sequence, limit),
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
        sequence=int(row["sequence"]),
        user_id=str(row["user_id"]),
        conversation_id=str(row["conversation_id"]),
        run_id=str(row["run_id"]),
        event_name=str(row["event_name"]),
        data=data,
    )


def next_sequence(
    connection: sqlite3.Connection,
    *,
    user_id: str,
    run_id: str,
) -> int:
    row = connection.execute(
        """
        SELECT COALESCE(MAX(sequence), 0) + 1 AS sequence
        FROM agent_stream_events
        WHERE user_id = ? AND run_id = ?
        """,
        (user_id, run_id),
    ).fetchone()
    return int(row["sequence"])
