from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from app.llm.graph.workflows.task_indexing.nodes.common import failure
from app.llm.subagents.task_indexing import TaskIndexingDelegation


async def initialize_task_indexing_node(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    call = state.get("active_tool_call")
    raw_request = call.get("args") if isinstance(call, Mapping) else None
    if not isinstance(raw_request, Mapping):
        return failure(
            code="INVALID_INDEXING_REQUEST",
            message="missing task indexing delegation",
            state=state,
        )

    try:
        request = TaskIndexingDelegation.model_validate(raw_request)
    except ValidationError as exc:
        return failure(
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
