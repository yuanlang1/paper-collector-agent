from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentRun, RunStatus


async def create(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    run_id: str,
    metadata: dict[str, Any],
) -> AgentRun:
    run = AgentRun(user_id=user_id, thread_id=thread_id, run_id=run_id, run_metadata=metadata)
    db.add(run)
    await db.flush()
    return run


async def get_by_run_id(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    run_id: str,
) -> AgentRun | None:
    return await db.scalar(
        select(AgentRun).where(
            AgentRun.user_id == user_id,
            AgentRun.thread_id == thread_id,
            AgentRun.run_id == run_id,
        )
    )


async def list_by_thread(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    limit: int = 20,
    offset: int = 0,
) -> list[AgentRun]:
    return list(
        await db.scalars(
            select(AgentRun)
            .where(AgentRun.user_id == user_id, AgentRun.thread_id == thread_id)
            .order_by(AgentRun.id.desc())
            .limit(limit)
            .offset(offset)
        )
    )


async def has_active(db: AsyncSession, *, user_id: int, thread_id: str) -> bool:
    return (
        await db.scalar(
            select(AgentRun.id)
            .where(
                AgentRun.user_id == user_id,
                AgentRun.thread_id == thread_id,
                AgentRun.status.in_((RunStatus.PENDING, RunStatus.RUNNING)),
            )
            .limit(1)
        )
        is not None
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
) -> AgentRun | None:
    values = {"status": status, "error_message": error_message}
    if status == RunStatus.RUNNING:
        values["started_at"] = func.now()
    else:
        values["finished_at"] = func.now()
    return await db.scalar(
        update(AgentRun)
        .where(
            AgentRun.user_id == user_id,
            AgentRun.thread_id == thread_id,
            AgentRun.run_id == run_id,
            AgentRun.status.in_(expected_status),
        )
        .values(**values)
        .returning(AgentRun)
        .execution_options(populate_existing=True)
    )
