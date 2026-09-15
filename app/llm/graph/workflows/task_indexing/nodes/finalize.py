from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.llm.graph.main.nodes.tool import build_action_result_update


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
