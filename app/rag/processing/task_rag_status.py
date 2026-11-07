from collections import Counter
from typing import Any

from app.infrastructure.paper_service.paper_client import (
    paper_service_client,
)


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
    papers = result["papers"]
    indexed_papers = [paper for paper in papers if paper["rag_status"] == "success"]
    counts = Counter(paper["rag_status"] for paper in papers)
    summary = normalize_rag_summary({
        "total": len(papers),
        "pending": counts["pending"],
        "indexing": counts["indexing"],
        "ready": len(indexed_papers),
        "skipped": counts["skipped"],
        "failed": counts["failed"],
        "ready_chunks": sum(paper["chunk_count"] for paper in indexed_papers),
    })
    rag_status = str(result["rag_status"] or "").upper()
    indexing_finished = (
        rag_status == "RAG_COMPLETED" and not summary["pending"] and not summary["indexing"]
    )
    return {
        "ok": True,
        "result": {
            "task_id": result["task_id"],
            "search_status": result["search_status"],
            "rag_status": result["rag_status"],
            "summary": summary,
            "is_rag_complete": indexing_finished and not summary["failed"],
            "can_generate_review": (
                indexing_finished
                and bool(indexed_papers)
                and all(
                    paper["chunk_count"] > 0
                    and paper.get("profile_status") == "success"
                    and paper.get("paper_profile")
                    for paper in indexed_papers
                )
            ),
        },
        "error": None,
        "metadata": response["metadata"],
    }
