from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from app.llm.graph.workflows.task_indexing.nodes.common import failure
from app.llm.subagents.task_indexing import TaskIndexingParams


async def initialize_task_indexing_node(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        request = TaskIndexingParams.model_validate(state)
    except ValidationError as exc:
        return failure(
            code="INVALID_INDEXING_REQUEST",
            message=f"invalid task indexing request: {exc}",
            state=state,
        )

    return {
        "task_id": request.task_id,
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
