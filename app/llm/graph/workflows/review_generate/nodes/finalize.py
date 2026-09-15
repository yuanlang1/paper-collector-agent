from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.llm.graph.main.nodes.tool import build_action_result_update


def failed(error: str, **details: Any) -> dict[str, Any]:
    return {
        **details,
        "stage": "failed",
        "status": "failed",
        "error": error,
    }


async def finalize_task_review_node(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    call = state.get("active_tool_call")
    if not isinstance(call, Mapping):
        raise RuntimeError("task review finalized without an active tool call")

    stage = state["stage"]
    final_review_artifact_ref = state.get("final_review_artifact_ref")
    reflection_report_artifact_ref = state.get("reflection_report_artifact_ref")
    study_extraction_report_artifact_ref = state.get(
        "study_extraction_report_artifact_ref"
    )

    if stage == "completed":
        available_artifacts = {
            name: artifact_ref
            for name, artifact_ref in {
                "corpus": state.get("corpus_artifact_ref"),
                "study_records": state.get("study_records_artifact_ref"),
                "framework": state.get("framework_artifact_ref"),
                "claims": state.get("claims_artifact_ref"),
                "evidence_ledger": state.get("evidence_ledger_artifact_ref"),
                "final_review": final_review_artifact_ref,
                "reflection_report": reflection_report_artifact_ref,
            }.items()
            if artifact_ref
        }
        result = {
            "status": "success",
            "summary": "学术综述已生成并通过反思审核。",
            "data": {
                "task_id": state["task_id"],
                "final_review_artifact_ref": final_review_artifact_ref,
                "available_artifacts": available_artifacts,
                "review_id": state.get("review_id"),
                "version_number": state.get("review_version_number"),
            },
            "artifact_refs": list(dict.fromkeys(available_artifacts.values())),
            "retryable": False,
            "error_code": None,
            "error_message": None,
        }
    elif stage == "blocked":
        result = {
            "status": "error",
            "summary": "RAG 尚未完成，暂时无法生成综述。",
            "data": {"task_id": state.get("task_id")},
            "artifact_refs": [],
            "retryable": True,
            "error_code": "RAG_NOT_READY",
            "error_message": None,
        }
    else:
        result = {
            "status": "error",
            "summary": "学术综述未能完成。",
            "data": {
                "task_id": state.get("task_id"),
                "study_extraction_report_artifact_ref": (
                    study_extraction_report_artifact_ref
                ),
                "warnings": list(state.get("warnings", [])),
            },
            "artifact_refs": list(
                dict.fromkeys(
                    artifact_ref
                    for artifact_ref in [
                        study_extraction_report_artifact_ref,
                        reflection_report_artifact_ref,
                    ]
                    if artifact_ref
                )
            ),
            "retryable": False,
            "error_code": stage,
            "error_message": state.get("error"),
        }

    return build_action_result_update(call=call, **result)
