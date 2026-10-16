from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentMessage
from app.db.repositories import agent_session


async def add(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    run_id: str,
    role: str,
    content: str | list[Any],
    metadata: dict[str, Any] | None = None,
) -> AgentMessage:
    messages = await add_many(
        db,
        user_id=user_id,
        thread_id=thread_id,
        messages=[
            {
                "run_id": run_id,
                "role": role,
                "content": content,
                "metadata": metadata or {},
            }
        ],
    )
    return messages[0]


async def add_many(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    messages: Sequence[dict[str, Any]],
) -> list[AgentMessage]:
    await agent_session.require_session(
        db,
        user_id=user_id,
        thread_id=thread_id,
    )
    rows = [
        AgentMessage(
            thread_id=thread_id,
            run_id=message["run_id"],
            role=message["role"],
            content=message["content"],
            message_metadata=message.get("metadata") or {},
            creator_id=user_id,
            modifier_id=user_id,
        )
        for message in messages
    ]

    db.add_all(rows)
    await db.flush()
    return rows


async def list_by_thread(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    before_id: int | None = None,
    limit: int = 50,
) -> list[AgentMessage]:
    await agent_session.require_session(
        db,
        user_id=user_id,
        thread_id=thread_id,
    )
    statement = select(AgentMessage).where(AgentMessage.thread_id == thread_id)
    if before_id is not None:
        statement = statement.where(AgentMessage.id < before_id)
    rows = list(
        await db.scalars(statement.order_by(AgentMessage.id.desc()).limit(limit))
    )
    return rows[::-1]


async def list_by_run(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    run_id: str,
) -> list[AgentMessage]:
    await agent_session.require_session(
        db,
        user_id=user_id,
        thread_id=thread_id,
    )

    return list(
        await db.scalars(
            select(AgentMessage)
            .where(
                AgentMessage.thread_id == thread_id,
                AgentMessage.run_id == run_id,
            )
            .order_by(AgentMessage.id)
        )
    )


async def list_by_range(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    start_id: int,
    end_id: int,
) -> list[AgentMessage]:
    if start_id > end_id:
        raise ValueError("start_id must not exceed end_id")
    await agent_session.require_session(
        db,
        user_id=user_id,
        thread_id=thread_id,
    )
    return list(
        await db.scalars(
            select(AgentMessage)
            .where(
                AgentMessage.thread_id == thread_id,
                AgentMessage.id.between(start_id, end_id),
            )
            .order_by(AgentMessage.id)
        )
    )
