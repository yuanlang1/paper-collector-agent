from collections import Counter
from typing import Any

from app.infrastructure.paper_service.paper_client import (
    paper_service_client,
)
from app.infrastructure.paper_service.types import TaskState


_SUMMARY_DEFAULTS = {
    "total": 0,
    "pending": 0,
    "indexing": 0,
    "ready": 0,
    "skipped": 0,
    "failed": 0,
    "ready_chunks": 0,
}


def normalize_rag_summary(value: dict[str, int] | None = None) -> dict[str, int]:
    return {
        key: int((value or {}).get(key, default))
        for key, default in _SUMMARY_DEFAULTS.items()
    }


async def get_task_rag_status(task_id: int) -> dict[str, Any]:
    response = await paper_service_client.get_task_review_papers(task_id)
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
