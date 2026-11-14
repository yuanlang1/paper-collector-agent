from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class AgentThreadSummaryRow:
    id: int
    user_id: int
    thread_id: str
    content: str
    end_message_id: str
    version: int
    creator_id: int
    creation_time: datetime
    modifier_id: int
    modification_time: datetime


_COLUMNS = """
    id, user_id, thread_id, content, end_message_id, version,
    creator_id, creation_time, modifier_id, modification_time
"""


async def get_by_thread_id(
    db: AsyncSession, *, user_id: int, thread_id: str
) -> AgentThreadSummaryRow | None:
    result = await db.execute(
        text(f"""
            SELECT {_COLUMNS}
            FROM agent_thread_summary
            WHERE user_id = :user_id AND thread_id = :thread_id
        """),
        dict(user_id=user_id, thread_id=thread_id),
    )
    row = result.mappings().one_or_none()
    return AgentThreadSummaryRow(**row) if row is not None else None


async def save(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    content: str,
    end_message_id: str,
    expected_version: int | None,
) -> AgentThreadSummaryRow | None:
    """Create for an owned thread, or update only the version the caller read."""
    params = dict(
        user_id=user_id,
        thread_id=thread_id,
        content=content,
        end_message_id=end_message_id,
    )
    if expected_version is None:
        statement = text(f"""
            INSERT INTO agent_thread_summary (user_id, thread_id, content, end_message_id)
            SELECT user_id, thread_id, :content, :end_message_id
            FROM agent_thread
            WHERE user_id = :user_id AND thread_id = :thread_id
            ON CONFLICT (user_id, thread_id) DO NOTHING
            RETURNING {_COLUMNS}
        """)
    else:
        params["expected_version"] = expected_version
        statement = text(f"""
            UPDATE agent_thread_summary
            SET content = :content,
                end_message_id = :end_message_id,
                version = version + 1,
                modification_time = now()
            WHERE user_id = :user_id
              AND thread_id = :thread_id
              AND version = :expected_version
            RETURNING {_COLUMNS}
        """)
    result = await db.execute(statement, params)
    row = result.mappings().one_or_none()
    return AgentThreadSummaryRow(**row) if row is not None else None
