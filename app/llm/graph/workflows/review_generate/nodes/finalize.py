from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.llm.subagents.task_review.contracts import TaskReviewHandoff


async def finalize_task_review_node(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    stage = state["stage"]
    final_review_artifact_ref = state.get("final_review_artifact_ref")
    reflection_report_artifact_ref = state.get(
        "reflection_report_artifact_ref"
    )

    if stage == "completed":
        handoff = TaskReviewHandoff(
            status="success",
            summary="学术综述已生成并通过反思审核。",
            data={
                "task_id": state["task_id"],
                "final_review_artifact_ref": final_review_artifact_ref,
                "upsert_review_handoff": state.get("handoff"),
            },
            artifact_refs=[
                artifact_ref
                for artifact_ref in [
                    final_review_artifact_ref,
                    reflection_report_artifact_ref,
                ]
                if artifact_ref
            ],
            retryable=False,
        )
    elif stage == "blocked":
        handoff = TaskReviewHandoff(
            status="error",
            summary="RAG 尚未完成，暂时无法生成综述。",
            data={"task_id": state.get("task_id")},
            artifact_refs=[],
            retryable=True,
            error_code="RAG_NOT_READY",
        )
    else:
        handoff = TaskReviewHandoff(
            status="error",
            summary="学术综述未能完成。",
            data={"task_id": state.get("task_id")},
            artifact_refs=[
                artifact_ref
                for artifact_ref in [reflection_report_artifact_ref]
                if artifact_ref
            ],
            retryable=False,
            error_code=stage,
            error_message=state.get("error"),
        )

    return {
        "task_review_handoff": handoff.model_dump(mode="json"),
    }
