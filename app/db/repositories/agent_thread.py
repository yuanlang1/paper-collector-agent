from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class AgentThreadRow:
    id: int
    thread_id: str
    user_id: int
    topic: str | None
    creator_id: int
    creation_time: datetime
    modifier_id: int
    modification_time: datetime


_COLUMNS = """
    id, thread_id, user_id, topic,
    creator_id, creation_time, modifier_id, modification_time
"""


async def create(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    topic: str | None = None,
) -> AgentThreadRow:
    result = await db.execute(
        text(f"""
            INSERT INTO agent_thread (user_id, thread_id, topic)
            VALUES (:user_id, :thread_id, :topic)
            RETURNING {_COLUMNS}
        """),
        dict(user_id=user_id, thread_id=thread_id, topic=topic),
    )
    return AgentThreadRow(**result.mappings().one())


async def get_by_thread_id(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    for_update: bool = False,
) -> AgentThreadRow | None:
    locking = " FOR UPDATE" if for_update else ""
    result = await db.execute(
        text(f"""
            SELECT {_COLUMNS}
            FROM agent_thread
            WHERE user_id = :user_id
              AND thread_id = :thread_id
        """ + locking),
        dict(user_id=user_id, thread_id=thread_id),
    )
    row = result.mappings().first()
    return AgentThreadRow(**row) if row is not None else None


async def touch(db: AsyncSession, *, user_id: int, thread_id: str) -> bool:
    return await db.scalar(
        text("""
            UPDATE agent_thread
            SET modification_time = now()
            WHERE user_id = :user_id
              AND thread_id = :thread_id
            RETURNING id
        """),
        dict(user_id=user_id, thread_id=thread_id),
    ) is not None


async def require_thread(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
) -> AgentThreadRow:
    thread = await get_by_thread_id(db, user_id=user_id, thread_id=thread_id)
    if thread is None:
        raise ValueError("Thread not found for user")
    return thread


async def list_by_user(
    db: AsyncSession,
    *,
    user_id: int,
    limit: int = 50,
    offset: int = 0,
) -> list[AgentThreadRow]:
    result = await db.execute(
        text(f"""
            SELECT {_COLUMNS}
            FROM agent_thread
            WHERE user_id = :user_id
            ORDER BY modification_time DESC, id DESC
            LIMIT :limit
            OFFSET :offset
        """),
        dict(user_id=user_id, limit=limit, offset=offset),
    )
    return [AgentThreadRow(**row) for row in result.mappings()]


async def update_topic(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    topic: str | None,
) -> AgentThreadRow:
    result = await db.execute(
        text(f"""
            UPDATE agent_thread
            SET modification_time = CASE
                    WHEN topic IS DISTINCT FROM CAST(:topic AS VARCHAR)
                      OR modifier_id IS DISTINCT FROM :user_id
                    THEN now()
                    ELSE modification_time
                END,
                topic = :topic,
                modifier_id = :user_id
            WHERE user_id = :user_id
              AND thread_id = :thread_id
            RETURNING {_COLUMNS}
        """),
        dict(user_id=user_id, thread_id=thread_id, topic=topic),
    )
    thread = result.mappings().first()
    if thread is None:
        raise ValueError("Thread not found for user")
    return AgentThreadRow(**thread)


async def delete(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
) -> None:
    # ponytail: identifiers come only from this fixed list, never request parameters.
    for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
        await db.execute(
            text(f"""
                DELETE FROM {table} AS checkpoint
                USING agent_thread AS thread
                WHERE checkpoint.thread_id = thread.thread_id
                  AND thread.user_id = :user_id
                  AND thread.thread_id = :thread_id
            """),
            dict(user_id=user_id, thread_id=thread_id),
        )
    for table in ("agent_run", "agent_thread"):
        await db.execute(
            text(f"""
                DELETE FROM {table}
                WHERE user_id = :user_id
                  AND thread_id = :thread_id
            """),
            dict(user_id=user_id, thread_id=thread_id),
        )
