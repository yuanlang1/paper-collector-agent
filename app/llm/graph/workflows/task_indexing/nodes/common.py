from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.infrastructure.grpc.task_service_grpc_client import TaskState
from app.rag.processing.task_rag_status import rag_progress_percent, task_status_value


INDEXABLE_STATES = {
    TaskState.SEARCH_COMPLETED.name,
    TaskState.RAG_RUNNING.name,
}


def failure(
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


def status_update(status: dict[str, Any]) -> dict[str, Any]:
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


def total_progress_payload(
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
