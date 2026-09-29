from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar


T = TypeVar("T")


class HistoryDatabase:
    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)

    def initialize(self) -> None:
        from app.history import chat_log, stream_events

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as connection:
            chat_log.create_schema(connection)
            stream_events.create_schema(connection)
            connection.execute("DROP TABLE IF EXISTS conversation_deletion_jobs")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
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

    async def run(
        self,
        operation: Callable[..., T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        return await asyncio.to_thread(operation, self, *args, **kwargs)


_history_database: HistoryDatabase | None = None


def initialize_history_database(db_path: str | Path) -> HistoryDatabase:
    global _history_database

    _history_database = HistoryDatabase(db_path)
    _history_database.initialize()
    return _history_database


def get_history_database() -> HistoryDatabase:
    if _history_database is None:
        raise RuntimeError("Chat history database has not been initialized")
    return _history_database
