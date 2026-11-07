import asyncio
import logging
from datetime import UTC, datetime, timedelta

from langchain_core.messages import BaseMessage, HumanMessage, convert_to_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import NotFoundException
from app.config import settings
from app.db.models import MemoryJobKind
from app.db.repositories import agent_run, agent_thread, memory_job
from app.db.repositories.memory_job import MemoryJobRow
from app.services.llm_setting_service import get_small_llm_config
from app.tasks.celery_app import get_worker_session_maker

logger = logging.getLogger(__name__)

LEASE_SECONDS = 300
HEARTBEAT_SECONDS = 60
MAX_ATTEMPTS = 3


class LeaseLostError(RuntimeError):
    pass


class MemorySourceError(ValueError):
    pass


async def load_messages(
    checkpointer, job: MemoryJobRow, *, after_id: str | None = None
) -> list[BaseMessage]:
    checkpoint = await checkpointer.aget_tuple(
        {
            "configurable": {
                "thread_id": job.thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": job.checkpoint_id,
            },
        }
    )

    if checkpoint is None:
        raise MemorySourceError("Memory checkpoint not found")

    messages = convert_to_messages(checkpoint.checkpoint["channel_values"]["messages"])
    end = next(
        (i for i, message in enumerate(messages) if message.id == job.end_id), None
    )
    if end is None:
        raise MemorySourceError("Memory end message not found in checkpoint")

    messages = messages[: end + 1]

    if job.kind == MemoryJobKind.EPISODE:
        if after_id is None:
            return messages

        start = next(
            (i for i, message in enumerate(messages) if message.id == after_id), None
        )

        if start is None:
            raise MemorySourceError(
                "Previous episode end message not found in checkpoint"
            )
        return messages[start + 1 :]

    start = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if isinstance(messages[i], HumanMessage)
        ),
        None,
    )

    if start is None:
        raise MemorySourceError("Memory run has no user message")

    return messages[start:]


async def claim_job(
    db: AsyncSession, job_id: str, kind: MemoryJobKind
) -> MemoryJobRow | None:
    async with db.begin():
        existing = await memory_job.get_by_job_id(db, job_id=job_id)
        if existing is None or existing.kind != kind:
            return None
        return await memory_job.claim(db, job_id=job_id, lease_seconds=LEASE_SECONDS)


async def validate_source(db: AsyncSession, job: MemoryJobRow) -> None:
    thread = await agent_thread.get_by_thread_id(
        db,
        user_id=job.user_id,
        thread_id=job.thread_id,
    )
    run = await agent_run.get_by_run_id(
        db,
        user_id=job.user_id,
        thread_id=job.thread_id,
        run_id=job.run_id,
    )
    if thread is None or run is None or run.status != "completed":
        raise MemorySourceError("Memory source is no longer an owned completed run")


async def load_job_messages(
    job: MemoryJobRow, *, after_id: str | None = None
) -> list[BaseMessage]:
    async with AsyncPostgresSaver.from_conn_string(
        settings.DATABASE_POSTGRES_URL
    ) as saver:
        return await load_messages(saver, job, after_id=after_id)


async def get_model(db: AsyncSession, user_id: int) -> ChatOpenAI:
    config = await get_small_llm_config(db, user_id=user_id)

    return ChatOpenAI(
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        streaming=False,
        extra_body={"thinking": {"type": "disabled"}},
    )


async def renew_lease(job: MemoryJobRow) -> None:
    factory = get_worker_session_maker()
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        async with factory.begin() as db:
            renewed = await memory_job.renew_lease(
                db,
                job_id=job.job_id,
                claim_token=job.claim_token,
                lease_seconds=LEASE_SECONDS,
            )
        if renewed is None:
            raise LeaseLostError("Memory job lease lost")


async def with_lease(job: MemoryJobRow, operation) -> bool:
    work = asyncio.create_task(operation)
    heartbeat = asyncio.create_task(renew_lease(job))
    try:
        done, _ = await asyncio.wait(
            (work, heartbeat), return_when=asyncio.FIRST_COMPLETED
        )
        if heartbeat in done:
            await heartbeat
        return await work
    finally:
        work.cancel()
        heartbeat.cancel()
        await asyncio.gather(work, heartbeat, return_exceptions=True)


async def finish_job(
    db: AsyncSession,
    job: MemoryJobRow,
    *,
    extracted: bool = True,
    error: Exception | None = None,
) -> dict[str, str]:
    if isinstance(error, LeaseLostError):
        return {"job_id": job.job_id, "status": "lease_lost"}
    async with db.begin():
        if isinstance(error, (NotFoundException, MemorySourceError)):
            updated = await memory_job.mark_failed(
                db,
                job_id=job.job_id,
                claim_token=job.claim_token,
                error_message=str(error)[:1000],
            )
        elif error is not None:
            updated = await memory_job.schedule_retry(
                db,
                job_id=job.job_id,
                claim_token=job.claim_token,
                available_at=datetime.now(UTC)
                + timedelta(seconds=30 * 2 ** (job.attempts - 1)),
                error_message=str(error)[:1000],
                max_attempts=MAX_ATTEMPTS,
            )
        else:
            finish = memory_job.complete if extracted else memory_job.mark_skipped
            updated = await finish(db, job_id=job.job_id, claim_token=job.claim_token)
    if error is not None:
        logger.warning(
            "Memory extraction failed: job_id=%s, error=%s",
            job.job_id,
            type(error).__name__,
        )
    return {"job_id": job.job_id, "status": updated.status if updated else "lease_lost"}
