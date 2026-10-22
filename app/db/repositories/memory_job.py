"""Memory job persistence for internal services; the caller owns every transaction."""

from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy import case, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentRun, MemoryJob, MemoryJobKind, MemoryJobStatus, RunStatus


async def _require_completed_run(
    db: AsyncSession, *, user_id: int, thread_id: str, run_id: str
) -> None:
    source = await db.scalar(
        select(AgentRun.id).where(
            AgentRun.user_id == user_id,
            AgentRun.thread_id == thread_id,
            AgentRun.run_id == run_id,
            AgentRun.status == RunStatus.COMPLETED,
        )
    )
    if source is None:
        raise ValueError("Completed run not found for user and thread")


async def create_run_jobs(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    run_id: str,
    checkpoint_id: str,
    end_id: str,
) -> list[MemoryJob]:
    """Create both per-run jobs in the caller's transaction; repeated calls reuse them."""
    await _require_completed_run(db, user_id=user_id, thread_id=thread_id, run_id=run_id)
    kinds = (MemoryJobKind.USERPROFILE, MemoryJobKind.FACT)
    await db.execute(
        insert(MemoryJob)
        .values([
            dict(
                job_id=f"job_{uuid4().hex}", kind=kind, user_id=user_id,
                thread_id=thread_id, run_id=run_id, checkpoint_id=checkpoint_id,
                end_id=end_id,
            )
            for kind in kinds
        ])
        .on_conflict_do_nothing(
            index_elements=[MemoryJob.run_id, MemoryJob.kind],
            index_where=MemoryJob.kind.in_(kinds),
        )
    )
    return list(await db.scalars(
        select(MemoryJob).where(
            MemoryJob.user_id == user_id, MemoryJob.thread_id == thread_id,
            MemoryJob.run_id == run_id, MemoryJob.kind.in_(kinds),
        ).order_by(MemoryJob.id)
    ))


async def create_episode_job(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    run_id: str,
    checkpoint_id: str,
    end_id: str,
    source_thread_update_time: datetime,
) -> MemoryJob | None:
    """The caller locks/checks the idle thread; duplicates or active episodes return None."""
    await _require_completed_run(db, user_id=user_id, thread_id=thread_id, run_id=run_id)
    return await db.scalar(
        insert(MemoryJob).values(
            job_id=f"job_{uuid4().hex}", kind=MemoryJobKind.EPISODE,
            user_id=user_id, thread_id=thread_id, run_id=run_id,
            checkpoint_id=checkpoint_id, end_id=end_id,
            source_thread_update_time=source_thread_update_time,
        ).on_conflict_do_nothing().returning(MemoryJob)
    )


async def get_by_job_id(db: AsyncSession, *, job_id: str) -> MemoryJob | None:
    """Internal worker lookup; job ownership is taken from this persisted row."""
    return await db.scalar(select(MemoryJob).where(MemoryJob.job_id == job_id))


async def list_dispatchable(db: AsyncSession, *, limit: int = 100) -> list[MemoryJob]:
    return list(await db.scalars(
        select(MemoryJob).where(
            MemoryJob.status == MemoryJobStatus.PENDING,
            MemoryJob.available_at <= func.clock_timestamp(),
        ).order_by(MemoryJob.available_at, MemoryJob.id).limit(limit)
    ))


async def claim(
    db: AsyncSession, *, job_id: str, lease_seconds: int
) -> MemoryJob | None:
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    return await db.scalar(
        update(MemoryJob).where(
            MemoryJob.job_id == job_id,
            MemoryJob.status == MemoryJobStatus.PENDING,
            MemoryJob.available_at <= func.clock_timestamp(),
        ).values(
            status=MemoryJobStatus.RUNNING,
            attempts=MemoryJob.attempts + 1,
            claim_token=uuid4().hex,
            lease_expires_at=func.clock_timestamp() + timedelta(seconds=lease_seconds),
        ).returning(MemoryJob).execution_options(populate_existing=True)
    )


def _owned_lease(job_id: str, claim_token: str):
    return (
        MemoryJob.job_id == job_id,
        MemoryJob.status == MemoryJobStatus.RUNNING,
        MemoryJob.claim_token == claim_token,
        MemoryJob.lease_expires_at > func.clock_timestamp(),
    )


async def renew_lease(
    db: AsyncSession, *, job_id: str, claim_token: str, lease_seconds: int
) -> MemoryJob | None:
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    return await db.scalar(
        update(MemoryJob).where(*_owned_lease(job_id, claim_token)).values(
            lease_expires_at=func.clock_timestamp() + timedelta(seconds=lease_seconds),
        ).returning(MemoryJob).execution_options(populate_existing=True)
    )


async def complete(db: AsyncSession, *, job_id: str, claim_token: str) -> MemoryJob | None:
    return await db.scalar(
        update(MemoryJob).where(*_owned_lease(job_id, claim_token)).values(
            status=MemoryJobStatus.COMPLETED, completed_at=func.clock_timestamp(),
            claim_token=None, lease_expires_at=None, error_message=None,
        ).returning(MemoryJob).execution_options(populate_existing=True)
    )


async def schedule_retry(
    db: AsyncSession,
    *,
    job_id: str,
    claim_token: str,
    available_at: datetime,
    error_message: str,
    max_attempts: int,
) -> MemoryJob | None:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    return await db.scalar(
        update(MemoryJob).where(*_owned_lease(job_id, claim_token)).values(
            status=case(
                (MemoryJob.attempts >= max_attempts, MemoryJobStatus.FAILED),
                else_=MemoryJobStatus.PENDING,
            ),
            available_at=available_at, error_message=error_message,
            claim_token=None, lease_expires_at=None,
        ).returning(MemoryJob).execution_options(populate_existing=True)
    )


async def mark_failed(
    db: AsyncSession, *, job_id: str, claim_token: str, error_message: str
) -> MemoryJob | None:
    return await db.scalar(
        update(MemoryJob).where(*_owned_lease(job_id, claim_token)).values(
            status=MemoryJobStatus.FAILED, error_message=error_message,
            claim_token=None, lease_expires_at=None,
        ).returning(MemoryJob).execution_options(populate_existing=True)
    )


async def mark_skipped(
    db: AsyncSession, *, job_id: str, claim_token: str
) -> MemoryJob | None:
    return await db.scalar(
        update(MemoryJob).where(*_owned_lease(job_id, claim_token)).values(
            status=MemoryJobStatus.SKIPPED, claim_token=None, lease_expires_at=None,
        ).returning(MemoryJob).execution_options(populate_existing=True)
    )


async def get_last_completed_episode(
    db: AsyncSession, *, user_id: int, thread_id: str
) -> MemoryJob | None:
    return await db.scalar(
        select(MemoryJob).where(
            MemoryJob.user_id == user_id, MemoryJob.thread_id == thread_id,
            MemoryJob.kind == MemoryJobKind.EPISODE,
            MemoryJob.status == MemoryJobStatus.COMPLETED,
        ).order_by(MemoryJob.completed_at.desc(), MemoryJob.id.desc()).limit(1)
    )


async def recover_expired_jobs(
    db: AsyncSession, *, max_attempts: int, limit: int = 100
) -> list[MemoryJob]:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    expired = select(MemoryJob.id).where(
        MemoryJob.status == MemoryJobStatus.RUNNING,
        MemoryJob.lease_expires_at <= func.clock_timestamp(),
    ).order_by(MemoryJob.id).limit(limit).with_for_update(skip_locked=True).cte("expired")
    return list(await db.scalars(
        update(MemoryJob).where(MemoryJob.id.in_(select(expired.c.id))).values(
            status=case(
                (MemoryJob.attempts >= max_attempts, MemoryJobStatus.FAILED),
                else_=MemoryJobStatus.PENDING,
            ),
            available_at=func.clock_timestamp(), error_message="Worker lease expired",
            claim_token=None, lease_expires_at=None,
        ).returning(MemoryJob).execution_options(populate_existing=True)
    ))
