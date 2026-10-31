import asyncio
import logging

from app.config import settings
from app.tasks.celery_app import celery_app, get_worker_loop

logger = logging.getLogger(__name__)


@celery_app.task(
    name="rag.index_batch",
    queue="rag",
    track_started=True,
    time_limit=settings.TASK_RAG_WORKFLOW_TIMEOUT_SECONDS + 60,
)
def index_batch(paper_ids: list[int], *, user_id: str) -> dict:
    async def execute():
        from app.services import rag_service

        async with asyncio.timeout(settings.TASK_RAG_WORKFLOW_TIMEOUT_SECONDS):
            return await rag_service.process_batch(paper_ids, user_id=user_id)

    logger.info("RAG batch started: paper_ids=%s", paper_ids)
    result = get_worker_loop().run_until_complete(execute())
    logger.info("RAG batch completed: paper_ids=%s", paper_ids)
    return result
