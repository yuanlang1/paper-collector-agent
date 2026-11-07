from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import RunStatus


@dataclass(frozen=True)
class AgentRunRow:
    id: int
    run_id: str
    thread_id: str
    user_id: int
    status: str
    run_metadata: dict[str, Any]
    started_at: datetime | None
    finished_at: datetime | None
    error_message: str | None
    creator_id: int
    creation_time: datetime
    modifier_id: int
    modification_time: datetime


_COLUMNS = """
    id, run_id, thread_id, user_id, status, metadata AS run_metadata,
    started_at, finished_at, error_message,
    creator_id, creation_time, modifier_id, modification_time
"""


async def create(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    run_id: str,
    metadata: dict[str, Any],
) -> AgentRunRow:
    result = await db.execute(
        text(f"""
            INSERT INTO agent_run (user_id, thread_id, run_id, metadata)
            VALUES (:user_id, :thread_id, :run_id, :metadata)
            RETURNING {_COLUMNS}
        """).bindparams(bindparam("metadata", type_=JSON)),
        {
            "user_id": user_id,
            "thread_id": thread_id,
            "run_id": run_id,
            "metadata": metadata,
        },
    )
    return AgentRunRow(**result.mappings().one())


async def get_by_run_id(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    run_id: str,
) -> AgentRunRow | None:
    result = await db.execute(
        text(f"""
            SELECT {_COLUMNS}
            FROM agent_run
            WHERE user_id = :user_id
              AND thread_id = :thread_id
              AND run_id = :run_id
        """),
        {"user_id": user_id, "thread_id": thread_id, "run_id": run_id},
    )
    row = result.mappings().first()
    return AgentRunRow(**row) if row is not None else None


async def list_by_thread(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    limit: int = 20,
    offset: int = 0,
) -> list[AgentRunRow]:
    result = await db.execute(
        text(f"""
            SELECT {_COLUMNS}
            FROM agent_run
            WHERE user_id = :user_id
              AND thread_id = :thread_id
            ORDER BY id DESC
            LIMIT :limit
            OFFSET :offset
        """),
        {"user_id": user_id, "thread_id": thread_id, "limit": limit, "offset": offset},
    )
    return [AgentRunRow(**row) for row in result.mappings()]


async def list_by_owned_thread(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    limit: int = 20,
    offset: int = 0,
) -> list[AgentRunRow] | None:
    result = await db.execute(
        text(f"""
            SELECT r.*
            FROM agent_thread AS t
            LEFT JOIN LATERAL (
                SELECT {_COLUMNS}
                FROM agent_run
                WHERE user_id = t.user_id
                  AND thread_id = t.thread_id
                ORDER BY id DESC
                LIMIT :limit
                OFFSET :offset
            ) AS r ON TRUE
            WHERE t.user_id = :user_id
              AND t.thread_id = :thread_id
            ORDER BY r.id DESC
        """),
        {"user_id": user_id, "thread_id": thread_id, "limit": limit, "offset": offset},
    )
    rows = list(result.mappings())
    if not rows:
        return None
    return [AgentRunRow(**row) for row in rows if row["id"] is not None]


async def has_active(db: AsyncSession, *, user_id: int, thread_id: str) -> bool:
    return bool(
        await db.scalar(
            text("""
                SELECT EXISTS (
                    SELECT 1
                    FROM agent_run
                    WHERE user_id = :user_id
                      AND thread_id = :thread_id
                      AND status IN ('pending', 'running')
                )
            """),
            {"user_id": user_id, "thread_id": thread_id},
        )
    )


async def update_status(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    run_id: str,
    status: RunStatus,
    expected_status: tuple[RunStatus, ...],
    error_message: str | None = None,
) -> AgentRunRow | None:
    result = await db.execute(
        text(f"""
            UPDATE agent_run
            SET status = :status,
                error_message = :error_message,
                started_at = CASE
                    WHEN CAST(:status AS VARCHAR) = 'running' THEN now()
                    ELSE started_at
                END,
                finished_at = CASE
                    WHEN CAST(:status AS VARCHAR) = 'running' THEN finished_at
                    ELSE now()
                END,
                modification_time = now()
            WHERE user_id = :user_id
              AND thread_id = :thread_id
              AND run_id = :run_id
              AND status IN :expected_status
            RETURNING {_COLUMNS}
        """).bindparams(bindparam("expected_status", expanding=True)),
        {
            "user_id": user_id,
            "thread_id": thread_id,
            "run_id": run_id,
            "status": str(status),
            "expected_status": [str(item) for item in expected_status],
            "error_message": error_message,
        },
    )
    row = result.mappings().first()
    return AgentRunRow(**row) if row is not None else None
