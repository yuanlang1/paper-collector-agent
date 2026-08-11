from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any
import uuid

from app.config import settings
from app.infrastructure.grpc.rag_grpc_client import RagGrpcClient, rag_grpc_client
from app.infrastructure.grpc.task_service_grpc_client import TaskState
from app.infrastructure.mineru.mineru_client import MinerUClient
from app.rag.data_preparation.data_preparation import DataPreparationModule
from app.rag.file_parser.file_parser import FileParser
from app.rag.index_construction.paper_content_index_construction import (
    PaperContentIndexConstructionModule,
)
from app.rag.index_construction.paper_index_construction import (
    PaperIndexConstructionModule,
)
from app.rag.processing.paper_rag_processor import PaperRagInput, PaperRagProcessor


logger = logging.getLogger(__name__)

DEFAULT_CLAIM_LIMIT = 20
DEFAULT_LEASE_SECONDS = 1800
_default_task_rag_batch_runner: TaskRagBatchRunner | None = None


class TaskRagBatchRunner:
    def __init__(
        self,
        *,
        processor: PaperRagProcessor | None = None,
        processor_factory: Callable[[], PaperRagProcessor] | None = None,
        client: RagGrpcClient | None = None,
    ) -> None:
        if processor is None and processor_factory is None:
            raise ValueError("processor or processor_factory is required")

        self.processor = processor
        self.processor_factory = processor_factory
        self.client = client or rag_grpc_client
        self._active_task_ids: set[int] = set()
        self._active_lock = asyncio.Lock()

    async def notify(
        self,
        task_id: int
    ) -> bool:
        async with self._active_lock:
            if task_id in self._active_task_ids:
                return False
            self._active_task_ids.add(task_id)

        asyncio.create_task(
            self._run_task(task_id),
            name = f"task-rag-{task_id}"
        )

        return True

    async def _run_task(
        self,
        task_id: int
    ) -> None:
        try:
            while True:
                logger.info("Task %s RAG worker start.", task_id)
                result = await self._run_once(task_id)

                if not result["ok"]:
                    logger.error(
                        "Task RAG batch stopped: task_id=%s, error=%s", task_id, result["error"]
                    )
                    return

                if (result["claimed_count"] == 0 or result["task_state"] != TaskState.RAG_RUNNING):
                    return
        except Exception:
            logger.exception("Task RAG worker crashed: task_id=%s", task_id)
        finally:
            async with self._active_lock:
                self._active_task_ids.discard(task_id)

    async def _run_once(
        self,
        task_id: int
    ) -> dict[str, Any]:
        batch_id = str(uuid.uuid4())
        claim = await self.client.claim_task_rag_papers(
            task_id = task_id,
            batch_id = batch_id,
            limit = DEFAULT_CLAIM_LIMIT,
            lease_seconds = DEFAULT_LEASE_SECONDS
        )

        if not claim["ok"]:
            return {
                "ok": False,
                "error": claim["error"]
            }

        claim_result = claim["result"]
        papers = claim_result["papers"]

        if not papers:
            return {
                "ok": True,
                "claimed_count": 0,
                "task_state": claim_result["task_state"],
            }

        inputs = [
            PaperRagInput(
                paper_id = paper["paper_id"],
                title = paper["title"],
                abstract = paper["paper_abstract"],
                pdf_url = paper["url"]
            )
            for paper in papers
        ]

        try:
            logger.info(
                "Task %s RAG processing start %s claim, batch id: %s",
                task_id,
                len(inputs),
                batch_id,
            )

            processed = await self._get_processor().process_many(inputs)
        except Exception as exc:
            logger.exception(
                "Task RAG processing failed; wait for lease recovery: task_id=%s, batch_id=%s", 
                task_id, 
                batch_id
            )

            return {
                "ok": False,
                "error": str(exc)
            }
        
        logger.info(
            "Task %s RAG processing end, start saved, papers: %s, batch_id: %s",
            task_id,
            len(processed),
            batch_id,
        )
        complete = await self.client.complete_task_rag_batch(
            task_id = task_id,
            batch_id = batch_id,
            results = [
                {
                    "paper_id": item.paper_id,
                    "status": item.status,
                    "chunk_count": item.chunk_count,
                    "error": item.error or ""
                }
                for item in processed
            ],
        )

        if not complete["ok"]:
            return {
                "ok": False,
                "error": complete["error"],
            }

        logger.info(
            "Task %s RAG processing end, saved, papers: %s, batch_id: %s, task state: %s",
            task_id,
            len(processed),
            batch_id,
            complete["result"]["task_state"],
        )

        return {
            "ok": True,
            "claimed_count": claim_result["claimed_count"],
            "task_state": complete["result"]["task_state"],
        }

    def _get_processor(self) -> PaperRagProcessor:
        if self.processor is None:
            self.processor = self.processor_factory()

        return self.processor


def get_task_rag_batch_runner() -> TaskRagBatchRunner:
    global _default_task_rag_batch_runner

    if _default_task_rag_batch_runner is None:
        _default_task_rag_batch_runner = TaskRagBatchRunner(
            processor_factory=_build_paper_rag_processor,
        )

    return _default_task_rag_batch_runner


def _build_paper_rag_processor() -> PaperRagProcessor:
    return PaperRagProcessor(
        file_parser=FileParser(
            mineru_client=MinerUClient(token=settings.MINERU_TOKEN),
        ),
        data_preparation=DataPreparationModule(),
        paper_index=PaperIndexConstructionModule(),
        paper_content_index=PaperContentIndexConstructionModule(),
    )


async def close_task_rag_batch_runner() -> None:
    global _default_task_rag_batch_runner

    runner = _default_task_rag_batch_runner
    _default_task_rag_batch_runner = None

    if runner is not None and runner.processor is not None:
        await runner.processor.file_parser.mineru_client.close()
