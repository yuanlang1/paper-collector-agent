import asyncio
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MemoryJobKind
from app.db.repositories import memory_job
from app.tasks.celery_app import celery_app
from app.tasks.celery_utils import run_async_task
from app.tasks.modules.memory import common
from app.tasks.modules.memory.extract_episode import extract_episode
from app.tasks.modules.memory.extract_fact import extract_fact
from app.tasks.modules.memory.extract_userprofile import extract_userprofile

logger = logging.getLogger(__name__)


async def scan_episodes(db: AsyncSession) -> int:
    created = 0
    for thread in await memory_job.lock_idle_threads(db):
        source = await memory_job.get_episode_source(
            db,
            user_id=thread.user_id,
            thread_id=thread.thread_id,
        )
        if source is None:
            continue
        job = await memory_job.create_episode_job(
            db,
            user_id=thread.user_id,
            thread_id=thread.thread_id,
            run_id=source.run_id,
            checkpoint_id=source.checkpoint_id,
            end_id=source.end_id,
            source_thread_update_time=thread.modification_time,
        )
        created += job is not None
    return created


async def maintain_jobs(db: AsyncSession) -> dict[str, int]:
    async with db.begin():
        recovered = await memory_job.recover_expired_jobs(
            db, max_attempts=common.MAX_ATTEMPTS
        )
        episodes = await scan_episodes(db)
    async with db.begin():
        pending = await memory_job.list_dispatchable(db)
    published = 0
    for job in pending:
        try:
            if job.kind == MemoryJobKind.USERPROFILE:
                await asyncio.to_thread(extract_userprofile.delay, job.job_id)
            elif job.kind == MemoryJobKind.FACT:
                await asyncio.to_thread(extract_fact.delay, job.job_id)
            elif job.kind == MemoryJobKind.EPISODE:
                await asyncio.to_thread(extract_episode.delay, job.job_id)
            published += 1
        except Exception:
            logger.warning(
                "Memory dispatch failed; pending job will be retried: job_id=%s",
                job.job_id,
            )
    return {"recovered": len(recovered), "episodes": episodes, "published": published}


@celery_app.task(bind=True, name="memory.maintain")
def maintain(self) -> dict[str, int]:
    return run_async_task(
        task_id=self.request.id,
        task_name=self.name,
        process_func=maintain_jobs,
    )
