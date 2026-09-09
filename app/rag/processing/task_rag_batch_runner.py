from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import inspect
import logging
from typing import Any, Literal
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
from app.rag.processing.task_rag_status import (
    empty_rag_summary,
    normalize_rag_summary,
    rag_progress_percent,
)


logger = logging.getLogger(__name__)

DEFAULT_CLAIM_LIMIT = 20
DEFAULT_LEASE_SECONDS = 1800
_default_task_rag_batch_runner: TaskRagBatchRunner | None = None


@dataclass(frozen=True, slots=True)
class RagError:
    code: str
    message: str
    batch_id: str | None = None
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class RagProgressEvent:
    type: Literal[
        "started",
        "claimed",
        "processing",
        "batch_committed",
        "completed",
        "failed",
    ]
    task_id: int
    batch_id: str | None
    sequence: int
    summary: dict[str, int]
    committed_summary: dict[str, int]
    batch_summary: dict[str, int]
    progress_percent: int
    committed_progress_percent: int
    error: RagError | None = None


@dataclass(frozen=True, slots=True)
class RagRunResult:
    ok: bool
    task_id: int
    task_state: str | None
    summary: dict[str, int]
    error: RagError | None = None


ProgressListener = Callable[[RagProgressEvent], Awaitable[None] | None]


@dataclass(slots=True)
class _TaskRagRun:
    task: asyncio.Task[RagRunResult] | None = None
    listeners: dict[int, ProgressListener] = field(default_factory=dict)
    next_listener_id: int = 0
    next_event_sequence: int = 0
    summary: dict[str, int] = field(default_factory=empty_rag_summary)
    committed_summary: dict[str, int] = field(default_factory=empty_rag_summary)


@dataclass(frozen=True, slots=True)
class _BatchRunResult:
    ok: bool
    claimed_count: int
    task_state: str | None
    summary: dict[str, int]
    error: RagError | None = None
    has_failed_papers: bool = False


class RagRunHandle:
    def __init__(
        self,
        *,
        task_id: int,
        started: bool,
        result: asyncio.Task[RagRunResult],
        runner: TaskRagBatchRunner,
        run: _TaskRagRun,
        listener_id: int | None,
    ) -> None:
        self.task_id = task_id
        self.started = started
        self.result = result
        self._runner = runner
        self._run = run
        self._listener_id = listener_id

    async def unsubscribe(self) -> None:
        if self._listener_id is None:
            return

        listener_id = self._listener_id
        self._listener_id = None
        await self._runner._remove_listener(
            task_id=self.task_id,
            run=self._run,
            listener_id=listener_id,
        )


async def wait_for_rag_run(
    handle: RagRunHandle,
    *,
    timeout_seconds: int,
) -> RagRunResult:
    try:
        return await asyncio.wait_for(
            asyncio.shield(handle.result),
            timeout=timeout_seconds,
        )
    finally:
        await handle.unsubscribe()


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
        self._runs: dict[int, _TaskRagRun] = {}
        self._runs_lock = asyncio.Lock()

    async def start_or_join(
        self,
        task_id: int,
        *,
        initial_summary: dict[str, int] | None = None,
        listener: ProgressListener | None = None,
    ) -> RagRunHandle:
        async with self._runs_lock:
            run = self._runs.get(task_id)
            started = run is None
            if run is None:
                summary = normalize_rag_summary(initial_summary)
                run = _TaskRagRun(
                    summary=summary,
                    committed_summary=dict(summary),
                )
                run.task = asyncio.create_task(
                    self._run_task(task_id, run),
                    name=f"task-rag-{task_id}",
                )
                self._runs[task_id] = run

            listener_id = self._add_listener(run, listener)
            assert run.task is not None
            return RagRunHandle(
                task_id=task_id,
                started=started,
                result=run.task,
                runner=self,
                run=run,
                listener_id=listener_id,
            )

    async def get_active_run(
        self,
        task_id: int,
        *,
        listener: ProgressListener | None = None,
    ) -> RagRunHandle | None:
        async with self._runs_lock:
            run = self._runs.get(task_id)
            if run is None or run.task is None:
                return None

            listener_id = self._add_listener(run, listener)
            return RagRunHandle(
                task_id=task_id,
                started=False,
                result=run.task,
                runner=self,
                run=run,
                listener_id=listener_id,
            )

    @staticmethod
    def _add_listener(
        run: _TaskRagRun,
        listener: ProgressListener | None,
    ) -> int | None:
        if listener is None:
            return None

        listener_id = run.next_listener_id
        run.next_listener_id += 1
        run.listeners[listener_id] = listener
        return listener_id

    async def _remove_listener(
        self,
        *,
        task_id: int,
        run: _TaskRagRun,
        listener_id: int,
    ) -> None:
        async with self._runs_lock:
            if self._runs.get(task_id) is run:
                run.listeners.pop(listener_id, None)

    async def _publish_progress(
        self,
        *,
        task_id: int,
        run: _TaskRagRun,
        event_type: RagProgressEvent.type,
        batch_id: str | None,
        batch_summary: dict[str, int] | None = None,
        error: RagError | None = None,
    ) -> None:
        async with self._runs_lock:
            run.next_event_sequence += 1
            event = RagProgressEvent(
                type=event_type,
                task_id=task_id,
                batch_id=batch_id,
                sequence=run.next_event_sequence,
                summary=dict(run.summary),
                committed_summary=dict(run.committed_summary),
                batch_summary=normalize_rag_summary(batch_summary),
                progress_percent=rag_progress_percent(run.summary),
                committed_progress_percent=rag_progress_percent(
                    run.committed_summary,
                ),
                error=error,
            )
            listeners = list(run.listeners.values())

        for listener in listeners:
            try:
                value = listener(event)
                if inspect.isawaitable(value):
                    await value
            except Exception:
                logger.warning(
                    "Task RAG progress listener failed: task_id=%s, event=%s",
                    task_id,
                    event.type,
                    exc_info=True,
                )

    async def _finish_run(
        self,
        *,
        task_id: int,
        run: _TaskRagRun,
        task_state: str | None,
        ok: bool,
        error: RagError | None = None,
    ) -> RagRunResult:
        await self._publish_progress(
            task_id=task_id,
            run=run,
            event_type="completed" if ok else "failed",
            batch_id=error.batch_id if error else None,
            error=error,
        )
        return RagRunResult(
            ok=ok,
            task_id=task_id,
            task_state=task_state,
            summary=run.summary,
            error=error,
        )

    async def _run_task(
        self,
        task_id: int,
        run: _TaskRagRun,
    ) -> RagRunResult:
        task_state: str | None = None

        try:
            logger.info("Task %s RAG worker start.", task_id)
            await self._publish_progress(
                task_id=task_id,
                run=run,
                event_type="started",
                batch_id=None,
            )

            while True:
                batch = await self._run_once(task_id, run)
                task_state = batch.task_state

                if not batch.ok:
                    assert batch.error is not None
                    logger.error(
                        "Task RAG batch stopped: task_id=%s, error=%s",
                        task_id,
                        batch.error.message,
                    )
                    return await self._finish_run(
                        task_id=task_id,
                        run=run,
                        task_state=task_state,
                        ok=False,
                        error=batch.error,
                    )

                if batch.has_failed_papers:
                    error = RagError(
                        code="RAG_PAPER_FAILED",
                        message="RAG 索引存在失败论文，已停止后续批次。",
                    )
                elif task_state == TaskState.RAG_RUNNING.name and batch.claimed_count > 0:
                    continue
                elif task_state == TaskState.RAG_COMPLETED.name:
                    return await self._finish_run(
                        task_id=task_id,
                        run=run,
                        task_state=task_state,
                        ok=True,
                    )
                elif task_state == TaskState.RAG_RUNNING.name:
                    error = RagError(
                        code="RAG_NO_CLAIMED_PAPERS",
                        message="RAG 仍在运行，但本 worker 未领取到可索引论文。",
                    )
                else:
                    error = RagError(
                        code="RAG_UNEXPECTED_TASK_STATE",
                        message=f"RAG worker 收到未预期的任务状态：{task_state or 'unknown'}。",
                    )

                return await self._finish_run(
                    task_id=task_id,
                    run=run,
                    task_state=task_state,
                    ok=False,
                    error=error,
                )

        except asyncio.CancelledError:
            return await self._finish_run(
                task_id=task_id,
                run=run,
                task_state=task_state,
                ok=False,
                error=RagError(
                    code="RAG_WORKER_CANCELLED",
                    message="RAG worker 已取消。",
                ),
            )
        except Exception as exc:
            logger.exception("Task RAG worker crashed: task_id=%s", task_id)
            return await self._finish_run(
                task_id=task_id,
                run=run,
                task_state=task_state,
                ok=False,
                error=RagError(
                    code="RAG_WORKER_CRASHED",
                    message=str(exc) or exc.__class__.__name__,
                ),
            )
        finally:
            async with self._runs_lock:
                if self._runs.get(task_id) is run:
                    self._runs.pop(task_id, None)

    async def _run_once(
        self,
        task_id: int,
        run: _TaskRagRun,
    ) -> _BatchRunResult:
        batch_id = str(uuid.uuid4())
        try:
            claim = await self.client.claim_task_rag_papers(
                task_id=task_id,
                batch_id=batch_id,
                limit=DEFAULT_CLAIM_LIMIT,
                lease_seconds=DEFAULT_LEASE_SECONDS,
            )
        except Exception as exc:
            return self._batch_failure(
                code="RAG_CLAIM_FAILED",
                message=str(exc) or exc.__class__.__name__,
                batch_id=batch_id,
                summary=run.summary,
            )

        if not claim.get("ok"):
            return self._batch_failure(
                code="RAG_CLAIM_FAILED",
                message=str(claim.get("error") or "Claim task RAG papers failed"),
                batch_id=batch_id,
                summary=run.summary,
            )

        claim_result = claim["result"]
        papers = claim_result["papers"]
        task_state = str(claim_result.get("task_state") or "")

        if not papers:
            return _BatchRunResult(
                ok=True,
                claimed_count=0,
                task_state=task_state,
                summary=run.summary,
            )

        batch_summary = empty_rag_summary()
        batch_summary["total"] = len(papers)
        batch_summary["indexing"] = len(papers)
        if run.summary["total"] == 0:
            run.summary["total"] = len(papers)
        run.summary["pending"] = max(0, run.summary["pending"] - len(papers))
        run.summary["indexing"] += len(papers)
        await self._publish_progress(
            task_id=task_id,
            run=run,
            event_type="claimed",
            batch_id=batch_id,
            batch_summary=batch_summary,
        )

        processed_count = 0

        async def on_paper_result(item) -> None:
            nonlocal processed_count
            processed_count += 1
            batch_summary["indexing"] = max(0, len(papers) - processed_count)
            batch_summary[item.status] += 1
            if item.status == "ready":
                batch_summary["ready_chunks"] += item.chunk_count
            run.summary["indexing"] = max(0, run.summary["indexing"] - 1)
            run.summary[item.status] += 1
            if item.status == "ready":
                run.summary["ready_chunks"] += item.chunk_count
            await self._publish_progress(
                task_id=task_id,
                run=run,
                event_type="processing",
                batch_id=batch_id,
                batch_summary=batch_summary,
            )

        try:
            inputs = [
                PaperRagInput(
                    paper_id=paper["paper_id"],
                    title=paper["title"],
                    abstract=paper["paper_abstract"],
                    pdf_url=paper["url"],
                )
                for paper in papers
            ]
            logger.info(
                "Task %s RAG processing start %s claim, batch id: %s",
                task_id,
                len(inputs),
                batch_id,
            )
            processed = await self._get_processor().process_many(
                inputs,
                on_paper_result=on_paper_result,
            )
        except Exception as exc:
            logger.exception(
                "Task RAG processing failed: task_id=%s, batch_id=%s",
                task_id,
                batch_id,
            )
            return await self._complete_failed_claim(
                task_id=task_id,
                batch_id=batch_id,
                papers=papers,
                error=exc,
                run=run,
            )

        try:
            complete = await self.client.complete_task_rag_batch(
                task_id=task_id,
                batch_id=batch_id,
                results=[
                    {
                        "paper_id": item.paper_id,
                        "status": item.status,
                        "chunk_count": item.chunk_count,
                        "error": item.error or "",
                    }
                    for item in processed
                ],
            )
        except Exception as exc:
            return self._batch_failure(
                code="RAG_BATCH_COMPLETE_FAILED",
                message=str(exc) or exc.__class__.__name__,
                batch_id=batch_id,
                claimed_count=len(papers),
                task_state=task_state,
                summary=run.summary,
            )

        if not complete.get("ok"):
            return self._batch_failure(
                code="RAG_BATCH_COMPLETE_FAILED",
                message=str(complete.get("error") or "Complete task RAG batch failed"),
                batch_id=batch_id,
                claimed_count=len(papers),
                task_state=task_state,
                summary=run.summary,
            )

        complete_result = complete["result"]
        summary = self._summary_from_complete(complete_result)
        run.summary = summary
        run.committed_summary = dict(summary)
        task_state = str(complete_result["task_state"])
        await self._publish_progress(
            task_id=task_id,
            run=run,
            event_type="batch_committed",
            batch_id=batch_id,
            batch_summary=batch_summary,
        )
        return _BatchRunResult(
            ok=True,
            claimed_count=int(claim_result["claimed_count"]),
            task_state=task_state,
            summary=summary,
            has_failed_papers=(
                any(item.status == "failed" for item in processed)
                or summary["failed"] > 0
            ),
        )

    async def _complete_failed_claim(
        self,
        *,
        task_id: int,
        batch_id: str,
        papers: list[dict[str, Any]],
        error: Exception,
        run: _TaskRagRun,
    ) -> _BatchRunResult:
        message = str(error) or error.__class__.__name__
        try:
            complete = await self.client.complete_task_rag_batch(
                task_id=task_id,
                batch_id=batch_id,
                results=[
                    {
                        "paper_id": paper["paper_id"],
                        "status": "failed",
                        "chunk_count": 0,
                        "error": message,
                    }
                    for paper in papers
                ],
            )
        except Exception as complete_error:
            return self._batch_failure(
                code="RAG_BATCH_COMPLETE_FAILED",
                message=(
                    f"RAG 处理失败：{message}；"
                    f"失败结果回写异常：{complete_error}"
                ),
                batch_id=batch_id,
                claimed_count=len(papers),
                summary=run.summary,
            )

        if not complete.get("ok"):
            return self._batch_failure(
                code="RAG_BATCH_COMPLETE_FAILED",
                message=(
                    f"RAG 处理失败：{message}；"
                    f"失败结果回写失败：{complete.get('error') or 'unknown error'}"
                ),
                batch_id=batch_id,
                claimed_count=len(papers),
                summary=run.summary,
            )

        complete_result = complete["result"]
        summary = self._summary_from_complete(complete_result)
        run.summary = summary
        run.committed_summary = dict(summary)
        await self._publish_progress(
            task_id=task_id,
            run=run,
            event_type="batch_committed",
            batch_id=batch_id,
            batch_summary=normalize_rag_summary({
                "total": len(papers),
                "failed": len(papers),
            }),
        )
        return self._batch_failure(
            code="RAG_PROCESSING_FAILED",
            message=f"RAG 解析或索引失败：{message}",
            batch_id=batch_id,
            claimed_count=len(papers),
            task_state=str(complete_result["task_state"]),
            summary=summary,
        )

    @staticmethod
    def _batch_failure(
        *,
        code: str,
        message: str,
        batch_id: str | None,
        claimed_count: int = 0,
        task_state: str | None = None,
        summary: dict[str, int] | None = None,
    ) -> _BatchRunResult:
        return _BatchRunResult(
            ok=False,
            claimed_count=claimed_count,
            task_state=task_state,
            summary=summary or empty_rag_summary(),
            error=RagError(
                code=code,
                message=message,
                batch_id=batch_id,
            ),
        )

    @staticmethod
    def _summary_from_complete(result: dict[str, Any]) -> dict[str, int]:
        return normalize_rag_summary({
            "total": int(result["total_count"]),
            "pending": int(result["pending_count"]),
            "indexing": int(result["indexing_count"]),
            "ready": int(result["ready_count"]),
            "skipped": int(result["skipped_count"]),
            "failed": int(result["failed_count"]),
            "ready_chunks": 0,
        })

    def _get_processor(self) -> PaperRagProcessor:
        if self.processor is None:
            assert self.processor_factory is not None
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
