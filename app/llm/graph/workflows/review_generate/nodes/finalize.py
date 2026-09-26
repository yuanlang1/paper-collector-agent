from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.llm.graph.main.nodes.tool import build_action_result_update


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

    stage = state["stage"]
    success = stage == "completed"
    blocked = stage == "blocked"
    artifacts = {
        key.removesuffix("_artifact_ref"): value
        for key, value in state.items()
        if key.endswith("_artifact_ref") and value
    }
    artifacts.update(
        {
            f"section_{key}": value
            for key, value in state.get("section_draft_artifact_refs", {}).items()
        }
    )
    if state.get("writing_review_plan_ref"):
        artifacts["writing_review_plan"] = state["writing_review_plan_ref"]
    summary = "学术综述已生成并通过反思审核。" if success else "学术综述未能完成。"
    if blocked:
        summary = "RAG 尚未完成，暂时无法生成综述。"
    elif state.get("error_code") in {
        "WRITING_REVISION_LIMIT_REACHED",
        "WRITING_REVIEW_BLOCKED",
    }:
        summary = "综述写作质量审核未通过，已保留草稿和审核记录。"
    return build_action_result_update(
        call=call,
        status="success" if success else "error",
        summary=summary,
        data={
            "task_id": state.get("task_id"),
            "available_artifacts": artifacts,
            "final_review_artifact_ref": state.get("final_review_artifact_ref"),
            "study_extraction_report_artifact_ref": state.get(
                "study_extraction_report_artifact_ref"
            ),
            "review_id": state.get("review_id"),
            "version_number": state.get("review_version_number"),
            "warnings": state.get("warnings", []),
        },
        artifact_refs=list(dict.fromkeys(artifacts.values())),
        retryable=False if success else blocked or state.get("retryable", False),
        error_code=None
        if success
        else "RAG_NOT_READY"
        if blocked
        else state.get("error_code", stage),
        error_message=state.get("error"),
    )
