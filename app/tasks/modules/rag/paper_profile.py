import asyncio

from app.config import settings
from app.schemas.rag import PostPaperProfile
from app.tasks.celery_app import celery_app
from app.tasks.celery_utils import run_async_task


@celery_app.task(
    bind=True,
    name="rag.extract_paper_profile",
    queue="rag",
    track_started=True,
    time_limit=settings.TASK_RAG_WORKFLOW_TIMEOUT_SECONDS + 60,
)
def extract_paper_profile_task(
    self, *, user_id: str, paper_id: str, oss_name: str
) -> dict[str, str]:
    async def execute(db):
        from app.services import rag_service

        request = PostPaperProfile(
            user_id=user_id, paper_id=paper_id, oss_name=oss_name
        )
        async with asyncio.timeout(settings.TASK_RAG_WORKFLOW_TIMEOUT_SECONDS):
            return await rag_service.extract_studies(db, **request.model_dump())

    return run_async_task(
        task_id=self.request.id,
        task_name=self.name,
        process_func=execute,
    )
