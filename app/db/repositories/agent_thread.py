from sqlalchemy import delete as sql_delete
from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentThread
from app.db.models.agent_run import AgentRun


async def create(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    topic: str | None = None,
) -> AgentThread:
    thread = AgentThread(
        user_id=user_id,
        thread_id=thread_id,
        topic=topic,
    )
    db.add(thread)
    await db.flush()
    return thread


async def get_by_thread_id(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    for_update: bool = False,
) -> AgentThread | None:
    statement = select(AgentThread).where(
        AgentThread.user_id == user_id,
        AgentThread.thread_id == thread_id,
    )
    if for_update:
        statement = statement.with_for_update()
    return await db.scalar(statement)


async def touch(db: AsyncSession, *, user_id: int, thread_id: str) -> bool:
    return (
        await db.scalar(
            update(AgentThread)
            .where(
                AgentThread.user_id == user_id,
                AgentThread.thread_id == thread_id,
            )
            .values(modification_time=func.now())
            .returning(AgentThread.id)
        )
        is not None
    )


async def require_thread(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
) -> AgentThread:
    thread = await get_by_thread_id(
        db,
        user_id=user_id,
        thread_id=thread_id,
    )
    if thread is None:
        raise ValueError("Thread not found for user")
    return thread


async def list_by_user(
    db: AsyncSession,
    *,
    user_id: int,
    limit: int = 50,
    offset: int = 0,
) -> list[AgentThread]:
    return list(
        await db.scalars(
            select(AgentThread)
            .where(AgentThread.user_id == user_id)
            .order_by(AgentThread.modification_time.desc(), AgentThread.id.desc())
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
) -> AgentThread:
    thread = await require_thread(
        db,
        user_id=user_id,
        thread_id=thread_id,
    )
    thread.topic = topic
    thread.modifier_id = user_id
    await db.flush()
    return thread


async def delete(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
) -> None:
    for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
        await db.execute(
            text(f"DELETE FROM {table} WHERE thread_id = :thread_id"),
            {"thread_id": thread_id},
        )

    for model in (AgentRun, AgentThread):
        await db.execute(
            sql_delete(model).where(
                model.user_id == user_id,
                model.thread_id == thread_id,
            )
        )

    await db.flush()
