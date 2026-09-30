from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from app.events.context import EventContext, current_event_context
from app.events.errors import EventPublicationError
from app.infrastructure.grpc.task_service_grpc_client import TaskState
from app.llm.graph.workflows.task_indexing.nodes.common import (
    failure,
    status_update,
    total_progress_payload,
)
from app.rag.processing.task_rag_batch_runner import (
    RagProgressEvent,
    get_task_rag_batch_runner,
    wait_for_rag_run,
)
from app.rag.processing.task_rag_status import (
    is_rag_failed,
    rag_progress_percent,
    wait_for_rag_terminal_status,
)


class RunOrWaitTaskRagNode:
    def __init__(self, runner=None) -> None:
        self.runner = runner or get_task_rag_batch_runner()

    async def __call__(
        self,
        state: Mapping[str, Any],
        config: Optional[RunnableConfig] = None,
    ) -> dict[str, Any]:
        task_id = state["task_id"]
        latest_event: RagProgressEvent | None = None
        event_context = current_event_context().scoped(
            source="subagent",
            workflow="task_indexing",
            node="run_or_wait",
        )
        publication_failure: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )

        async def on_rag_event(event: RagProgressEvent) -> None:
            nonlocal latest_event
            latest_event = event
            try:
                await event_context.bus.publish(
                    event_context.event(
                        "subagent_progress",
                        total_progress_payload(
                            task_id=event.task_id,
                            summary=event.summary,
                            progress_percent=event.progress_percent,
                            batch_id=event.batch_id,
                            event_type=event.type,
                            committed_summary=event.committed_summary,
                            committed_progress_percent=(
                                event.committed_progress_percent
                            ),
                            batch_summary=event.batch_summary,
                        ),
                    ),
                )
            except EventPublicationError as exc:
                if not publication_failure.done():
                    publication_failure.set_exception(exc)
                raise

        if state["remote_task_state"] == TaskState.SEARCH_COMPLETED.name:
            handle = await self.runner.start_or_join(
                task_id,
                initial_summary=state["overall_summary"],
                listener=on_rag_event,
            )
        else:
            handle = await self.runner.get_active_run(
                task_id,
                listener=on_rag_event,
            )
            if handle is None:
                return await self._wait_for_remote_task(state, event_context)

        try:
            worker_result = await self._wait_for_result_or_publication_failure(
                handle=handle,
                publication_failure=publication_failure,
                timeout_seconds=state["timeout_seconds"],
            )
        except TimeoutError:
            return failure(
                code="RAG_INDEX_TIMEOUT",
                message="等待 RAG 索引完成超时，后台索引仍在继续。",
                state=state,
                timed_out=True,
            )
        finally:
            await handle.unsubscribe()

        event_update = self._event_update(latest_event)
        worker = {
            "task_state": worker_result.task_state,
            "summary": worker_result.summary,
            "error": (
                {
                    "code": worker_result.error.code,
                    "message": worker_result.error.message,
                    "batch_id": worker_result.error.batch_id,
                }
                if worker_result.error is not None
                else None
            ),
        }
        if not worker_result.ok:
            return {
                **event_update,
                "worker_result": worker,
                **failure(
                    code=worker_result.error.code,
                    message=worker_result.error.message,
                    state={**state, **event_update},
                ),
            }
        return {
            **event_update,
            "worker_result": worker,
            "stage": "verifying",
            "status": "running",
        }

    async def _wait_for_remote_task(
        self,
        state: Mapping[str, Any],
        event_context: EventContext,
    ) -> dict[str, Any]:
        async def on_status(status: dict[str, Any]) -> None:
            summary = status["result"]["summary"]
            await event_context.bus.publish(
                event_context.event(
                    "subagent_progress",
                    total_progress_payload(
                        task_id=state["task_id"],
                        summary=summary,
                        progress_percent=rag_progress_percent(summary),
                        event_type="remote_status",
                    ),
                ),
            )

        status = await wait_for_rag_terminal_status(
            task_id=state["task_id"],
            poll_interval_seconds=state["poll_interval_seconds"],
            timeout_seconds=state["timeout_seconds"],
            on_status=on_status,
        )
        update = status_update(status) if status.get("ok") else {}
        if not status.get("ok"):
            return {
                **update,
                **failure(
                    code=str(status.get("error") or "RAG_STATUS_FAILED"),
                    message="等待远端 RAG 索引状态失败。",
                    state={**state, **update},
                    timed_out=status.get("error") == "RAG_INDEX_TIMEOUT",
                ),
            }
        if is_rag_failed(status):
            return {
                **update,
                **failure(
                    code="RAG_INDEX_FAILED",
                    message="远端 RAG 索引失败。",
                    state={**state, **update},
                ),
            }
        return {**update, "stage": "verifying", "status": "running"}

    @staticmethod
    async def _wait_for_result_or_publication_failure(
        *,
        handle,
        publication_failure: asyncio.Future[None],
        timeout_seconds: int,
    ):
        waiter = asyncio.create_task(
            wait_for_rag_run(handle, timeout_seconds=timeout_seconds),
        )
        done, _ = await asyncio.wait(
            {waiter, publication_failure},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if publication_failure in done:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
            return publication_failure.result()
        return await waiter

    @staticmethod
    def _event_update(event: RagProgressEvent | None) -> dict[str, Any]:
        if event is None:
            return {}
        return {
            "overall_summary": event.summary,
            "committed_summary": event.committed_summary,
            "current_batch_summary": event.batch_summary,
            "overall_progress_percent": event.progress_percent,
            "committed_progress_percent": event.committed_progress_percent,
        }
