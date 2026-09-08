import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable
from time import monotonic
from typing import Any

from app.infrastructure.grpc.paper_service_grpc_client import (
    paper_service_grpc_client,
)
from app.infrastructure.grpc.task_service_grpc_client import TaskState


StatusListener = Callable[[dict[str, Any]], Awaitable[None]]

_FAILED_TASK_STATES = {
    TaskState.SEARCH_FAILED.name,
    TaskState.CANCELLED.name,
    TaskState.RAG_FAILED.name,
}
_SUMMARY_DEFAULTS = {
    "total": 0,
    "pending": 0,
    "indexing": 0,
    "ready": 0,
    "skipped": 0,
    "failed": 0,
    "ready_chunks": 0,
}


def empty_rag_summary() -> dict[str, int]:
    return dict(_SUMMARY_DEFAULTS)


def normalize_rag_summary(value: dict[str, int] | None = None) -> dict[str, int]:
    return {
        key: int((value or {}).get(key, default))
        for key, default in _SUMMARY_DEFAULTS.items()
    }


def rag_progress_percent(summary: dict[str, int]) -> int:
    total = int(summary.get("total", 0))
    if total <= 0:
        return 0
    completed = sum(
        int(summary.get(key, 0))
        for key in ("ready", "skipped", "failed")
    )
    return min(100, round(completed / total * 100))


async def get_task_rag_status(task_id: int) -> dict[str, Any]:
    response = await paper_service_grpc_client.get_task_review_papers(task_id)
    if not response["ok"]:
        return response

    result = response["result"]
    counts = Counter(paper["rag_status"] for paper in result["papers"])
    summary = normalize_rag_summary({
        "total": len(result["papers"]),
        "pending": counts["pending"],
        "indexing": counts["indexing"],
        "ready": counts["ready"],
        "skipped": counts["skipped"],
        "failed": counts["failed"],
        "ready_chunks": sum(
            paper["chunk_count"]
            for paper in result["papers"]
            if paper["rag_status"] == "ready"
        ),
    })
    task_status = str(result["task_status"] or "").upper()
    return {
        "ok": True,
        "result": {
            "task_id": result["task_id"],
            "task_status": result["task_status"],
            "summary": summary,
            "is_rag_complete": (
                task_status == TaskState.RAG_COMPLETED.name
                and not summary["pending"]
                and not summary["indexing"]
                and not summary["failed"]
            ),
            "can_generate_review": (
                task_status == TaskState.RAG_COMPLETED.name
                and not summary["pending"]
                and not summary["indexing"]
                and not summary["failed"]
                and bool(summary["ready"])
            ),
        },
        "error": None,
        "metadata": response["metadata"],
    }


def task_status_value(status: dict[str, Any]) -> str:
    return str(status["result"]["task_status"] or "").upper()


def is_rag_failed(status: dict[str, Any]) -> bool:
    return (
        task_status_value(status) in _FAILED_TASK_STATES
        or bool(status["result"]["summary"]["failed"])
    )


async def wait_for_rag_terminal_status(
    *,
    task_id: int,
    poll_interval_seconds: int,
    timeout_seconds: int,
    on_status: StatusListener | None = None,
) -> dict[str, Any]:
    deadline = monotonic() + timeout_seconds
    latest: dict[str, Any] | None = None

    while True:
        status = await get_task_rag_status(task_id)
        if not status.get("ok"):
            return status

        latest = status
        if on_status is not None:
            await on_status(status)

        if is_rag_failed(status) or status["result"]["is_rag_complete"]:
            return status

        remaining_seconds = deadline - monotonic()
        if remaining_seconds <= 0:
            return {
                "ok": False,
                "result": latest["result"],
                "error": "RAG_INDEX_TIMEOUT",
                "metadata": latest.get("metadata"),
            }

        await asyncio.sleep(min(poll_interval_seconds, remaining_seconds))
