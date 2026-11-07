from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.state import TaskReviewOutputState


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


async def finalize_task_review_node(state: Mapping[str, Any],) -> TaskReviewOutputState:
    stage = state["stage"]
    success = stage == "completed"
    blocked = stage == "blocked"
    if success:
        try:
            await LocalArtifactStore().delete_run_directories(
                (state["child_run_id"],)
            )
        except OSError:
            logger.warning("Task-review artifact cleanup failed.", exc_info=True)
    summary = "学术综述已生成并通过反思审核。" if success else "学术综述未能完成。"
    if blocked:
        summary = "RAG 尚未完成，暂时无法生成综述。"
    elif state.get("error_code") in {
        "WRITING_REVISION_LIMIT_REACHED",
        "WRITING_REVIEW_BLOCKED",
    }:
        summary = "综述写作质量审核未通过。"
    lines = [summary]
    if state.get("task_id") is not None:
        lines.append(f"检索任务 ID：{state['task_id']}。")
    if state.get("review_id") is not None:
        lines.append(f"综述 ID：{state['review_id']}，版本：{state.get('review_version_number', 1)}。")
    if state.get("error"):
        code = "RAG_NOT_READY" if blocked else state.get("error_code") or stage
        lines.append(f"失败原因（{code}）：{state['error']}")
    if not success and state.get("claim_verification_artifact_ref"):
        lines.append(f"论点核验报告：{state['claim_verification_artifact_ref']}")
    lines.extend(f"警告：{warning}" for warning in state.get("warnings", []))
    return {
        "content": "\n".join(lines),
        "status": "success" if success else "timed_out" if stage == "timed_out" else "failed",
    }
