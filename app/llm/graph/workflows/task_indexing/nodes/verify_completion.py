from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.llm.graph.workflows.task_indexing.nodes.common import failure, status_update
from app.rag.processing.task_rag_status import get_task_rag_status, is_rag_failed


async def verify_task_indexing_node(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    status = await get_task_rag_status(state["task_id"])
    if not status.get("ok"):
        return failure(
            code="RAG_STATUS_FAILED",
            message="索引完成后无法确认 RAG 状态。",
            state=state,
        )

    update = status_update(status)
    if status["result"]["is_rag_complete"]:
        return {**update, "stage": "completed", "status": "completed"}
    return {
        **update,
        **failure(
            code=("RAG_INDEX_FAILED" if is_rag_failed(status) else "RAG_NOT_COMPLETE"),
            message=(
                "RAG 索引失败。"
                if is_rag_failed(status)
                else "RAG worker 已结束，但远端未确认索引完成。"
            ),
            state={**state, **update},
        ),
    }
