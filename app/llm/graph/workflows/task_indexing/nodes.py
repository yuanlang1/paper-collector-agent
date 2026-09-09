from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from app.infrastructure.grpc.task_service_grpc_client import TaskState
from app.llm.graph.main.nodes.tool import build_action_result_update
from app.llm.streaming.notify import Notifier, langgraph_notifier
from app.llm.subagents.task_indexing import TaskIndexingDelegation
from app.rag.processing.task_rag_batch_runner import (
    RagProgressEvent,
    get_task_rag_batch_runner,
    wait_for_rag_run,
)
from app.rag.processing.task_rag_status import (
    get_task_rag_status,
    is_rag_failed,
    rag_progress_percent,
    task_status_value,
    wait_for_rag_terminal_status,
)


_INDEXABLE_STATES = {
    TaskState.SEARCH_COMPLETED.name,
    TaskState.RAG_RUNNING.name,
}


def _status_update(status: dict[str, Any]) -> dict[str, Any]:
    result = status["result"]
    summary = dict(result["summary"])
    progress = rag_progress_percent(summary)
    return {
        "rag_status": result,
        "remote_task_state": task_status_value(status),
        "overall_summary": summary,
        "committed_summary": dict(summary),
        "current_batch_summary": {},
        "overall_progress_percent": progress,
        "committed_progress_percent": progress,
    }


def _total_progress_payload(
    *,
    task_id: int,
    summary: Mapping[str, int],
    progress_percent: int,
    batch_id: str | None = None,
    event_type: str | None = None,
    committed_summary: Mapping[str, int] | None = None,
    committed_progress_percent: int | None = None,
    batch_summary: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    completed = sum(int(summary.get(key, 0)) for key in ("ready", "skipped", "failed"))
    batch = dict(batch_summary or {})
    batch_completed = sum(int(batch.get(key, 0)) for key in ("ready", "skipped", "failed"))
    message = f"知识库索引：{completed}/{summary.get('total', 0)} 篇（{progress_percent}%）"
    if batch.get("total"):
        message += f"；当前批次：{batch_completed}/{batch['total']} 篇"

    return {
        "progress": progress_percent,
        "progress_percent": progress_percent,
        "phase": "index",
        "phase_label": "索引论文到知识库",
        "status": "running",
        "task_id": task_id,
        "message": message,
        "data": {
            "batch_id": batch_id,
            "event_type": event_type,
            "summary": dict(summary),
            "committed_summary": dict(committed_summary or summary),
            "batch_summary": batch,
            "committed_progress_percent": (
                progress_percent
                if committed_progress_percent is None
                else committed_progress_percent
            ),
        },
    }


def _failure(
    *,
    code: str,
    message: str,
    state: Mapping[str, Any],
    timed_out: bool = False,
) -> dict[str, Any]:
    return {
        "stage": "timed_out" if timed_out else "failed",
        "status": "timed_out" if timed_out else "failed",
        "error_code": code,
        "error": message,
        "overall_summary": state.get("overall_summary", {}),
        "committed_summary": state.get("committed_summary", {}),
        "current_batch_summary": state.get("current_batch_summary", {}),
    }


async def initialize_task_indexing_node(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    call = state.get("active_tool_call")
    raw_request = call.get("args") if isinstance(call, Mapping) else None
    if not isinstance(raw_request, Mapping):
        return _failure(
            code="INVALID_INDEXING_REQUEST",
            message="missing task indexing delegation",
            state=state,
        )

    try:
        request = TaskIndexingDelegation.model_validate(raw_request)
    except ValidationError as exc:
        return _failure(
            code="INVALID_INDEXING_REQUEST",
            message=f"invalid task indexing request: {exc}",
            state=state,
        )

    return {
        "task_id": request.task_id,
        "timeout_seconds": request.timeout_seconds,
        "poll_interval_seconds": request.poll_interval_seconds,
        "stage": "checking_status",
        "status": "running",
        "error_code": None,
        "error": None,
        "rag_status": None,
        "overall_summary": {},
        "committed_summary": {},
        "current_batch_summary": {},
        "overall_progress_percent": 0,
        "committed_progress_percent": 0,
        "worker_result": None,
        "warnings": [],
    }


async def check_task_rag_status_node(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    status = await get_task_rag_status(state["task_id"])
    if not status.get("ok"):
        return _failure(
            code="RAG_STATUS_FAILED",
            message=str(status.get("error") or "RAG 状态查询失败。"),
            state=state,
        )

    update = _status_update(status)
    if status["result"]["is_rag_complete"]:
        return {**update, "stage": "completed", "status": "completed"}
    if is_rag_failed(status):
        return {
            **update,
            **_failure(
                code="RAG_INDEX_FAILED",
                message="RAG 索引已失败，无法保存到知识库。",
                state={**state, **update},
            ),
        }
    if task_status_value(status) not in _INDEXABLE_STATES:
        return {
            **update,
            **_failure(
                code="TASK_NOT_READY_FOR_RAG",
                message="任务尚未完成论文检索，无法启动知识库索引。",
                state={**state, **update},
            ),
        }
    return {**update, "stage": "indexing", "status": "running"}


class RunOrWaitTaskRagNode:
    def __init__(self, runner=None) -> None:
        self.runner = runner or get_task_rag_batch_runner()

    async def __call__(
        self,
        state: Mapping[str, Any],
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        task_id = state["task_id"]
        latest_event: RagProgressEvent | None = None
        notify = langgraph_notifier(config).scoped(
            source="subagent",
            workflow="task_indexing",
            node="run_or_wait",
        )

        def on_rag_event(event: RagProgressEvent) -> None:
            nonlocal latest_event
            latest_event = event
            notify(
                "subagent_progress",
                _total_progress_payload(
                    task_id=event.task_id,
                    summary=event.summary,
                    progress_percent=event.progress_percent,
                    batch_id=event.batch_id,
                    event_type=event.type,
                    committed_summary=event.committed_summary,
                    committed_progress_percent=event.committed_progress_percent,
                    batch_summary=event.batch_summary,
                ),
            )

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
                return await self._wait_for_remote_task(state, notify)

        try:
            worker_result = await wait_for_rag_run(
                handle,
                timeout_seconds=state["timeout_seconds"],
            )
        except TimeoutError:
            return _failure(
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
                **_failure(
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
        notify: Notifier,
    ) -> dict[str, Any]:
        async def on_status(status: dict[str, Any]) -> None:
            summary = status["result"]["summary"]
            notify(
                "subagent_progress",
                _total_progress_payload(
                    task_id=state["task_id"],
                    summary=summary,
                    progress_percent=rag_progress_percent(summary),
                    event_type="remote_status",
                ),
            )

        status = await wait_for_rag_terminal_status(
            task_id=state["task_id"],
            poll_interval_seconds=state["poll_interval_seconds"],
            timeout_seconds=state["timeout_seconds"],
            on_status=on_status,
        )
        update = _status_update(status) if status.get("ok") else {}
        if not status.get("ok"):
            return {
                **update,
                **_failure(
                    code=str(status.get("error") or "RAG_STATUS_FAILED"),
                    message="等待远端 RAG 索引状态失败。",
                    state={**state, **update},
                    timed_out=status.get("error") == "RAG_INDEX_TIMEOUT",
                ),
            }
        if is_rag_failed(status):
            return {
                **update,
                **_failure(
                    code="RAG_INDEX_FAILED",
                    message="远端 RAG 索引失败。",
                    state={**state, **update},
                ),
            }
        return {**update, "stage": "verifying", "status": "running"}

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


async def verify_task_indexing_node(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    status = await get_task_rag_status(state["task_id"])
    if not status.get("ok"):
        return _failure(
            code="RAG_STATUS_FAILED",
            message="索引完成后无法确认 RAG 状态。",
            state=state,
        )

    update = _status_update(status)
    if status["result"]["is_rag_complete"]:
        return {**update, "stage": "completed", "status": "completed"}
    return {
        **update,
        **_failure(
            code=("RAG_INDEX_FAILED" if is_rag_failed(status) else "RAG_NOT_COMPLETE"),
            message=(
                "RAG 索引失败。"
                if is_rag_failed(status)
                else "RAG worker 已结束，但远端未确认索引完成。"
            ),
            state={**state, **update},
        ),
    }


async def finalize_task_indexing_node(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    call = state.get("active_tool_call")
    if not isinstance(call, Mapping):
        raise RuntimeError("task indexing finalized without an active tool call")

    data = {
        "task_id": state.get("task_id"),
        "overall_summary": state.get("overall_summary", {}),
        "committed_summary": state.get("committed_summary", {}),
        "overall_progress_percent": state.get("overall_progress_percent", 0),
        "committed_progress_percent": state.get("committed_progress_percent", 0),
        "worker_result": state.get("worker_result"),
    }
    if state.get("stage") == "completed":
        return build_action_result_update(
            call=call,
            status="success",
            summary="任务论文已全部保存到知识库。",
            data=data,
            artifact_refs=[],
            retryable=False,
            error_code=None,
            error_message=None,
        )

    error_code = str(state.get("error_code") or "TASK_INDEXING_FAILED")
    retryable = state.get("stage") == "timed_out"
    return build_action_result_update(
        call=call,
        status="error",
        summary="知识库索引未完成。",
        data=data,
        artifact_refs=[],
        retryable=retryable,
        error_code=error_code,
        error_message=state.get("error"),
    )
