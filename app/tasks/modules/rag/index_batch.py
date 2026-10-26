import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import asdict

from app.config import settings
from app.schemas.rag import IndexBatchRequest
from app.tasks.celery_app import celery_app, get_worker_loop

logger = logging.getLogger(__name__)


async def process_batch(paper_ids: list[int]) -> dict:
    from app.infrastructure.nacos_registry import NacosRegistry
    from app.infrastructure.paper_service.http_client import PaperServiceHttpClient
    from app.infrastructure.paper_service.paper_client import PaperServiceClient

    paper_ids = IndexBatchRequest(paper_ids=paper_ids).paper_ids

    async with AsyncExitStack() as resources:
        registry = NacosRegistry()
        if not settings.PAPER_SERVICE_BASE_URL:
            await registry.start_discovery()
            resources.push_async_callback(registry.client.shutdown)

        http = PaperServiceHttpClient(registry=registry)
        await http.start()
        resources.push_async_callback(http.close)

        response = await PaperServiceClient(http).batch_get_paper_details(paper_ids)
        if not response["ok"]:
            raise RuntimeError(response["error"])

        results = {
            paper_id: {
                "paper_id": paper_id,
                "status": "failed",
                "chunk_count": 0,
                "error": "PAPER_NOT_FOUND",
            }
            for paper_id in response["result"]["missing_paper_ids"]
        }
        papers = response["result"]["papers"]
        if papers:
            from app.infrastructure.mineru.mineru_client import MinerUClient
            from app.rag.data_preparation.data_preparation import DataPreparationModule
            from app.rag.file_parser.file_parser import FileParser
            from app.rag.processing.paper_rag_processor import (
                PaperRagInput,
                PaperRagProcessor,
            )

            mineru = MinerUClient(
                token=settings.MINERU_TOKEN,
                parse_timeout=settings.MINERU_PARSE_TIMEOUT_SECONDS,
            )
            resources.push_async_callback(mineru.close)

            processor = PaperRagProcessor(
                file_parser=FileParser(mineru_client=mineru),
                data_preparation=DataPreparationModule(),
            )
            processed = await processor.process_many(
                [PaperRagInput(**paper) for paper in papers]
            )
            results.update((item.paper_id, asdict(item)) for item in processed)
        return {"results": [results[paper_id] for paper_id in paper_ids]}


@celery_app.task(
    name="rag.index_batch",
    queue="rag",
    track_started=True,
    time_limit=settings.TASK_RAG_WORKFLOW_TIMEOUT_SECONDS + 60,
)
def index_batch(paper_ids: list[int]) -> dict:
    async def execute():
        async with asyncio.timeout(settings.TASK_RAG_WORKFLOW_TIMEOUT_SECONDS):
            return await process_batch(paper_ids)

    logger.info("RAG batch started: paper_ids=%s", paper_ids)
    result = get_worker_loop().run_until_complete(execute())
    logger.info("RAG batch completed: paper_ids=%s", paper_ids)
    return result
