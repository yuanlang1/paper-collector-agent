from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MemoryJobKind
from app.db.repositories import memory_job
from app.memory.manager import create_episode_manager
from app.memory.store import open_memory_store
from app.tasks.celery_app import celery_app
from app.tasks.celery_utils import run_async_task
from app.tasks.modules.memory import common


async def _extract_episode(db: AsyncSession, job_id: str) -> dict[str, str]:
    job = await common.claim_job(db, job_id, MemoryJobKind.EPISODE)
    if job is None:
        return {"job_id": job_id, "status": "ignored"}

    async def extract() -> bool:
        async with db.begin():
            await common.validate_source(db, job)
            if not await memory_job.is_episode_current(db, job=job):
                return False
            previous = await memory_job.get_last_completed_episode(
                db,
                user_id=job.user_id,
                thread_id=job.thread_id,
            )
        messages = await common.load_job_messages(
            job,
            after_id=previous.end_id if previous is not None else None,
        )
        if not messages:
            return True
        model = await common.get_model(db, job.user_id)
        async with open_memory_store() as store:
            manager = create_episode_manager(
                model=model,
                store=store,
                user_id=str(job.user_id),
                thread_id=job.thread_id,
                job_id=job.job_id,
            )
            await manager.ainvoke({"messages": messages})
        return True

    try:
        extracted = await common.with_lease(job, extract())
    except Exception as exc:
        return await common.finish_job(db, job, error=exc)
    return await common.finish_job(db, job, extracted=extracted)


@celery_app.task(bind=True, name="memory.extract_episode", acks_late=True)
def extract_episode(self, job_id: str) -> dict[str, str]:
    return run_async_task(
        task_id=self.request.id,
        task_name=self.name,
        process_func=lambda db: _extract_episode(db, job_id),
    )
