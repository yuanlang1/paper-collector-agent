from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentSession


async def create(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    topic: str | None = None,
) -> AgentSession:
    session = AgentSession(
        user_id=user_id,
        thread_id=thread_id,
        topic=topic,
        creator_id=user_id,
        modifier_id=user_id,
    )
    db.add(session)
    await db.flush()
    return session


async def get_by_thread_id(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
) -> AgentSession | None:
    return await db.scalar(
        select(AgentSession).where(
            AgentSession.user_id == user_id,
            AgentSession.thread_id == thread_id,
        )
    )


async def require_session(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
) -> AgentSession:
    session = await get_by_thread_id(
        db,
        user_id=user_id,
        thread_id=thread_id,
    )
    if session is None:
        raise ValueError("Session not found for user")
    return session


async def list_by_user(
    db: AsyncSession,
    *,
    user_id: int,
    limit: int = 50,
    offset: int = 0,
) -> list[AgentSession]:
    return list(
        await db.scalars(
            select(AgentSession)
            .where(AgentSession.user_id == user_id)
            .order_by(AgentSession.modification_time.desc(), AgentSession.id.desc())
            .limit(limit)
            .offset(offset)
        )
    )


async def update_topic(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    topic: str | None,
) -> AgentSession:
    session = await require_session(
        db,
        user_id=user_id,
        thread_id=thread_id,
    )
    session.topic = topic
    session.modifier_id = user_id
    await db.flush()
    return session
