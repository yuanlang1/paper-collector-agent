from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.main.nodes.tool import build_action_result_update


logger = logging.getLogger(__name__)


def failed(error: str, **details: Any) -> dict[str, Any]:
    return {
        **details,
        "stage": "failed",
        "status": "failed",
        "error_code": details.get("error_code", "REVIEW_FAILED"),
        "retryable": details.get("retryable", False),
        "error": error,
    }


async def finalize_task_review_node(state: Mapping[str, Any],) -> dict[str, Any]:
    call = state.get("active_tool_call")
    if not isinstance(call, Mapping):
        raise RuntimeError("task review finalized without an active tool call")

    try:
        await LocalArtifactStore().delete_run_directories(
            (state["child_run_id"],)
        )
    except OSError:
        logger.warning("Task-review artifact cleanup failed.", exc_info=True)

    stage = state["stage"]
    success = stage == "completed"
    blocked = stage == "blocked"
    summary = "学术综述已生成并通过反思审核。" if success else "学术综述未能完成。"
    if blocked:
        summary = "RAG 尚未完成，暂时无法生成综述。"
    elif state.get("error_code") in {
        "WRITING_REVISION_LIMIT_REACHED",
        "WRITING_REVIEW_BLOCKED",
    }:
        summary = "综述写作质量审核未通过。"
    return build_action_result_update(
        call=call,
        status="success" if success else "error",
        summary=summary,
        data={
            "task_id": state.get("task_id"),
            "review_id": state.get("review_id"),
            "version_number": state.get("review_version_number"),
            "warnings": state.get("warnings", []),
        },
        artifact_refs=[],
        retryable=False if success else blocked or state.get("retryable", False),
        error_code=None
        if success
        else "RAG_NOT_READY"
        if blocked
        else state.get("error_code", stage),
        error_message=state.get("error"),
    )
