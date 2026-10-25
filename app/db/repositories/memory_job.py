"""Memory job persistence for internal services; the caller owns every transaction."""

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.agent_thread import AgentThreadRow


@dataclass(frozen=True)
class MemoryJobRow:
    id: int
    job_id: str
    kind: str
    user_id: int
    thread_id: str
    run_id: str
    checkpoint_id: str
    end_id: str
    source_thread_update_time: datetime | None
    status: str
    attempts: int
    available_at: datetime
    lease_expires_at: datetime | None
    claim_token: str | None
    completed_at: datetime | None
    error_message: str | None
    creator_id: int
    creation_time: datetime
    modifier_id: int
    modification_time: datetime


_COMPLETED_SOURCE = """
    SELECT r.user_id, r.thread_id, r.run_id
    FROM agent_run AS r
    JOIN agent_thread AS t
      ON t.thread_id = r.thread_id
     AND t.user_id = r.user_id
    WHERE r.user_id = :user_id
      AND r.thread_id = :thread_id
      AND r.run_id = :run_id
      AND r.status = 'completed'
"""

_OWNED_LEASE = """
    job_id = :job_id
    AND status = 'running'
    AND claim_token = :claim_token
    AND lease_expires_at > clock_timestamp()
"""

_EPISODE_CANDIDATE = """
    FROM agent_thread AS thread
    JOIN LATERAL (
        SELECT run_id, status
        FROM agent_run
        WHERE user_id = thread.user_id
          AND thread_id = thread.thread_id
        ORDER BY id DESC
        LIMIT 1
    ) AS latest ON TRUE
    JOIN memory_job AS source
      ON source.user_id = thread.user_id
     AND source.thread_id = thread.thread_id
     AND source.run_id = latest.run_id
     AND source.kind = 'userprofile'
    WHERE thread.modification_time < clock_timestamp() - interval '10 minutes'
      AND latest.status = 'completed'
      AND NOT EXISTS (
          SELECT 1
          FROM memory_job AS episode
          WHERE episode.user_id = thread.user_id
            AND episode.thread_id = thread.thread_id
            AND episode.kind = 'episode'
            AND (
                episode.status IN ('pending', 'running')
                OR episode.source_thread_update_time = thread.modification_time
                OR (episode.status = 'completed' AND episode.end_id = source.end_id)
            )
      )
"""


async def lock_idle_threads(
    db: AsyncSession, *, limit: int = 100
) -> list[AgentThreadRow]:
    result = await db.execute(
        text(f"""
            SELECT thread.*
            {_EPISODE_CANDIDATE}
            ORDER BY thread.modification_time, thread.id
            LIMIT :limit
            FOR UPDATE OF thread SKIP LOCKED
        """),
        {"limit": limit},
    )
    return [AgentThreadRow(**row) for row in result.mappings()]


async def get_episode_source(
    db: AsyncSession, *, user_id: int, thread_id: str
) -> MemoryJobRow | None:
    # Recheck in a new statement after locking the thread for a fresh snapshot.
    result = await db.execute(
        text(f"""
            SELECT source.*
            {_EPISODE_CANDIDATE}
              AND thread.user_id = :user_id
              AND thread.thread_id = :thread_id
        """),
        {"user_id": user_id, "thread_id": thread_id},
    )
    row = result.mappings().first()
    return MemoryJobRow(**row) if row is not None else None


async def is_episode_current(db: AsyncSession, *, job: MemoryJobRow) -> bool:
    return bool(await db.scalar(
        text("""
            SELECT EXISTS (
                SELECT 1
                FROM agent_thread AS thread
                JOIN LATERAL (
                    SELECT run_id, status
                    FROM agent_run
                    WHERE user_id = thread.user_id
                      AND thread_id = thread.thread_id
                    ORDER BY id DESC
                    LIMIT 1
                ) AS latest ON TRUE
                WHERE thread.user_id = :user_id
                  AND thread.thread_id = :thread_id
                  AND thread.modification_time = :source_thread_update_time
                  AND thread.modification_time < clock_timestamp() - interval '10 minutes'
                  AND latest.run_id = :run_id
                  AND latest.status = 'completed'
            )
        """),
        {
            "user_id": job.user_id,
            "thread_id": job.thread_id,
            "run_id": job.run_id,
            "source_thread_update_time": job.source_thread_update_time,
        },
    ))


async def create_run_jobs(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    run_id: str,
    checkpoint_id: str,
    end_id: str,
) -> list[MemoryJobRow]:
    params = {
        "user_id": user_id,
        "thread_id": thread_id,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "end_id": end_id,
        "profile_job_id": f"job_{uuid4().hex}",
        "fact_job_id": f"job_{uuid4().hex}",
    }
    result = await db.execute(
        text(f"""
            WITH source AS (
                {_COMPLETED_SOURCE}
            ),
            inserted AS (
                INSERT INTO memory_job (
                    job_id, kind, user_id, thread_id, run_id, checkpoint_id, end_id
                )
                SELECT jobs.job_id,
                       jobs.kind,
                       source.user_id,
                       source.thread_id,
                       source.run_id,
                       :checkpoint_id,
                       :end_id
                FROM source
                CROSS JOIN (
                    VALUES
                        (:profile_job_id, 'userprofile'),
                        (:fact_job_id, 'fact')
                ) AS jobs(job_id, kind)
                ON CONFLICT (run_id, kind)
                    WHERE kind IN ('userprofile', 'fact')
                DO NOTHING
                RETURNING id
            )
            SELECT EXISTS(SELECT 1 FROM source) AS source_valid,
                   (SELECT count(*) FROM inserted) AS inserted_count
        """),
        params,
    )
    if not result.mappings().one()["source_valid"]:
        raise ValueError("Completed run not found for user and thread")

    result = await db.execute(
        text("""
            SELECT job.*
            FROM memory_job AS job
            JOIN agent_thread AS thread
              ON thread.thread_id = job.thread_id
             AND thread.user_id = job.user_id
            WHERE job.user_id = :user_id
              AND job.thread_id = :thread_id
              AND job.run_id = :run_id
              AND job.kind IN ('userprofile', 'fact')
            ORDER BY job.id
        """),
        params,
    )
    return [MemoryJobRow(**row) for row in result.mappings()]


async def create_episode_job(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    run_id: str,
    checkpoint_id: str,
    end_id: str,
    source_thread_update_time: datetime,
) -> MemoryJobRow | None:
    result = await db.execute(
        text(f"""
            WITH source AS (
                {_COMPLETED_SOURCE}
            ),
            inserted AS (
                INSERT INTO memory_job (
                    job_id, kind, user_id, thread_id, run_id, checkpoint_id, end_id,
                    source_thread_update_time
                )
                SELECT :job_id,
                       'episode',
                       user_id,
                       thread_id,
                       run_id,
                       :checkpoint_id,
                       :end_id,
                       :source_thread_update_time
                FROM source
                ON CONFLICT DO NOTHING
                RETURNING *
            )
            SELECT inserted.*,
                   EXISTS(SELECT 1 FROM source) AS source_valid
            FROM (VALUES (1)) AS requested(value)
            LEFT JOIN inserted ON TRUE
        """),
        dict(
            job_id=f"job_{uuid4().hex}",
            user_id=user_id,
            thread_id=thread_id,
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            end_id=end_id,
            source_thread_update_time=source_thread_update_time,
        ),
    )
    row = dict(result.mappings().one())
    if not row.pop("source_valid"):
        raise ValueError("Completed run not found for user and thread")
    return MemoryJobRow(**row) if row["job_id"] is not None else None


async def get_by_job_id(db: AsyncSession, *, job_id: str) -> MemoryJobRow | None:
    result = await db.execute(
        text("""
            SELECT *
            FROM memory_job
            WHERE job_id = :job_id
        """),
        {"job_id": job_id},
    )
    row = result.mappings().first()
    return MemoryJobRow(**row) if row is not None else None


async def list_dispatchable(
    db: AsyncSession, *, limit: int = 100
) -> list[MemoryJobRow]:
    result = await db.execute(
        text("""
            SELECT *
            FROM memory_job
            WHERE status = 'pending'
              AND available_at <= clock_timestamp()
            ORDER BY available_at, id
            LIMIT :limit
        """),
        {"limit": limit},
    )
    return [MemoryJobRow(**row) for row in result.mappings()]


async def claim(
    db: AsyncSession, *, job_id: str, lease_seconds: int
) -> MemoryJobRow | None:
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    result = await db.execute(
        text("""
            UPDATE memory_job
            SET status = 'running',
                attempts = attempts + 1,
                claim_token = :claim_token,
                lease_expires_at = clock_timestamp() + :lease_seconds * interval '1 second',
                modification_time = now()
            WHERE job_id = :job_id
              AND status = 'pending'
              AND available_at <= clock_timestamp()
            RETURNING *
        """),
        dict(job_id=job_id, claim_token=uuid4().hex, lease_seconds=lease_seconds),
    )
    row = result.mappings().first()
    return MemoryJobRow(**row) if row is not None else None


async def renew_lease(
    db: AsyncSession, *, job_id: str, claim_token: str, lease_seconds: int
) -> MemoryJobRow | None:
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    result = await db.execute(
        text(f"""
            UPDATE memory_job
            SET lease_expires_at = clock_timestamp() + :lease_seconds * interval '1 second',
                modification_time = now()
            WHERE {_OWNED_LEASE}
            RETURNING *
        """),
        dict(job_id=job_id, claim_token=claim_token, lease_seconds=lease_seconds),
    )
    row = result.mappings().first()
    return MemoryJobRow(**row) if row is not None else None


async def complete(
    db: AsyncSession, *, job_id: str, claim_token: str
) -> MemoryJobRow | None:
    result = await db.execute(
        text(f"""
            UPDATE memory_job
            SET status = 'completed',
                completed_at = clock_timestamp(),
                claim_token = NULL,
                lease_expires_at = NULL,
                error_message = NULL,
                modification_time = now()
            WHERE {_OWNED_LEASE}
            RETURNING *
        """),
        dict(job_id=job_id, claim_token=claim_token),
    )
    row = result.mappings().first()
    return MemoryJobRow(**row) if row is not None else None


async def schedule_retry(
    db: AsyncSession,
    *,
    job_id: str,
    claim_token: str,
    available_at: datetime,
    error_message: str,
    max_attempts: int,
) -> MemoryJobRow | None:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    result = await db.execute(
        text(f"""
            UPDATE memory_job
            SET status = CASE
                    WHEN attempts >= :max_attempts THEN 'failed'
                    ELSE 'pending'
                END,
                available_at = :available_at,
                error_message = :error_message,
                claim_token = NULL,
                lease_expires_at = NULL,
                modification_time = now()
            WHERE {_OWNED_LEASE}
            RETURNING *
        """),
        dict(
            job_id=job_id,
            claim_token=claim_token,
            available_at=available_at,
            error_message=error_message,
            max_attempts=max_attempts,
        ),
    )
    row = result.mappings().first()
    return MemoryJobRow(**row) if row is not None else None


async def mark_failed(
    db: AsyncSession, *, job_id: str, claim_token: str, error_message: str
) -> MemoryJobRow | None:
    result = await db.execute(
        text(f"""
            UPDATE memory_job
            SET status = 'failed',
                error_message = :error_message,
                claim_token = NULL,
                lease_expires_at = NULL,
                modification_time = now()
            WHERE {_OWNED_LEASE}
            RETURNING *
        """),
        dict(job_id=job_id, claim_token=claim_token, error_message=error_message),
    )
    row = result.mappings().first()
    return MemoryJobRow(**row) if row is not None else None


async def mark_skipped(
    db: AsyncSession, *, job_id: str, claim_token: str
) -> MemoryJobRow | None:
    result = await db.execute(
        text(f"""
            UPDATE memory_job
            SET status = 'skipped',
                claim_token = NULL,
                lease_expires_at = NULL,
                modification_time = now()
            WHERE {_OWNED_LEASE}
            RETURNING *
        """),
        dict(job_id=job_id, claim_token=claim_token),
    )
    row = result.mappings().first()
    return MemoryJobRow(**row) if row is not None else None


async def get_last_completed_episode(
    db: AsyncSession, *, user_id: int, thread_id: str
) -> MemoryJobRow | None:
    result = await db.execute(
        text("""
            SELECT *
            FROM memory_job
            WHERE user_id = :user_id
              AND thread_id = :thread_id
              AND kind = 'episode'
              AND status = 'completed'
            ORDER BY completed_at DESC, id DESC
            LIMIT 1
        """),
        dict(user_id=user_id, thread_id=thread_id),
    )
    row = result.mappings().first()
    return MemoryJobRow(**row) if row is not None else None


async def recover_expired_jobs(
    db: AsyncSession, *, max_attempts: int, limit: int = 100
) -> list[MemoryJobRow]:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    result = await db.execute(
        text("""
            WITH expired AS (
                SELECT id
                FROM memory_job
                WHERE status = 'running'
                  AND lease_expires_at <= clock_timestamp()
                ORDER BY id
                LIMIT :limit
                FOR UPDATE SKIP LOCKED
            )
            UPDATE memory_job AS job
            SET status = CASE
                    WHEN job.attempts >= :max_attempts THEN 'failed'
                    ELSE 'pending'
                END,
                available_at = clock_timestamp(),
                error_message = 'Worker lease expired',
                claim_token = NULL,
                lease_expires_at = NULL,
                modification_time = now()
            FROM expired
            WHERE job.id = expired.id
            RETURNING job.*
        """),
        dict(max_attempts=max_attempts, limit=limit),
    )
    return [MemoryJobRow(**row) for row in result.mappings()]
