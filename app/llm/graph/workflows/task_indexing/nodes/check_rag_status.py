from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.llm.graph.workflows.task_indexing.nodes.common import (
    INDEXABLE_STATES,
    failure,
    status_update,
)
from app.rag.processing.task_rag_status import (
    get_task_rag_status,
    is_rag_failed,
    task_status_value,
)


async def check_task_rag_status_node(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    status = await get_task_rag_status(state["task_id"])
    if not status.get("ok"):
        return failure(
            code="RAG_STATUS_FAILED",
            message=str(status.get("error") or "RAG 状态查询失败。"),
            state=state,
        )

    update = status_update(status)
    if status["result"]["is_rag_complete"]:
        return {**update, "stage": "completed", "status": "completed"}
    if is_rag_failed(status):
        return {
            **update,
            **failure(
                code="RAG_INDEX_FAILED",
                message="RAG 索引已失败，无法保存到知识库。",
                state={**state, **update},
            ),
        }
    if task_status_value(status) not in INDEXABLE_STATES:
        return {
            **update,
            **failure(
                code="TASK_NOT_READY_FOR_RAG",
                message="任务尚未完成论文检索，无法启动知识库索引。",
                state={**state, **update},
            ),
        }
    return {**update, "stage": "indexing", "status": "running"}
