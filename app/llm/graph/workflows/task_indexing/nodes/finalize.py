from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.main.nodes.tool import build_action_result_update


logger = logging.getLogger(__name__)


async def finalize_task_indexing_node(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    call = state.get("active_tool_call")
    if not isinstance(call, Mapping):
        raise RuntimeError("task indexing finalized without an active tool call")

    try:
        await LocalArtifactStore().delete_run_directories(
            (state["child_run_id"],)
        )
    except OSError:
        logger.warning("Task-indexing artifact cleanup failed.", exc_info=True)

    data = {
        "task_id": state.get("task_id"),
        "overall_summary": state.get("overall_summary", {}),
        "committed_summary": state.get("committed_summary", {}),
        "overall_progress_percent": state.get("overall_progress_percent", 0),
        "committed_progress_percent": state.get("committed_progress_percent", 0),
        "worker_result": state.get("worker_result"),
    }
    if state.get("stage") == "completed":
        ready = int(data["overall_summary"].get("ready", 0))
        skipped = int(data["overall_summary"].get("skipped", 0))
        summary = f"知识库索引完成：{ready} 篇论文可用于全文检索。"
        if skipped:
            summary += f"已跳过 {skipped} 篇。"
        return build_action_result_update(
            call=call,
            status="success",
            summary=summary,
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
