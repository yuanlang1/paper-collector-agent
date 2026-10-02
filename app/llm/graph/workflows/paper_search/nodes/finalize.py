from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.main.nodes.tool import build_action_result_update


logger = logging.getLogger(__name__)


async def finalize_paper_search_node(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    call = state.get("active_tool_call")
    if not isinstance(call, Mapping):
        raise RuntimeError("paper search finalized without an active tool call")

    try:
        await LocalArtifactStore().delete_run_directories(
            (state["child_run_id"],)
        )
    except OSError:
        logger.warning("Paper-search artifact cleanup failed.", exc_info=True)

    stage = str(state.get("stage"))
    progress = state.get("progress") or {}
    status = {
        "completed": "success",
        "partial_failed": "partial",
    }.get(stage, "error")

    if stage == "completed":
        summary = (
            f"论文检索完成：推荐 {progress.get('recommendation_kept', 0)} 篇，"
            f"保存关系 {progress.get('persistence_saved', 0)} 条。"
        )
    elif stage == "partial_failed":
        summary = "论文检索部分完成，请查看警告与结果。"
    else:
        summary = "论文检索未完成。"

    return build_action_result_update(
        call=call,
        status=status,
        summary=summary,
        data={
            "workflow_stage": stage,
            "search_task_id": state.get("paper_service_task_id"),
            "counts": {
                "discovered": progress.get("discovered", 0),
                "deduplicated": progress.get("deduplicated", 0),
                "existing_papers": progress.get("existing_in_database", 0),
                "new_papers": progress.get("new_candidates", 0),
                "recommended_papers": progress.get("recommendation_kept", 0),
                "persisted_relations": progress.get("persistence_saved", 0),
            },
            "supplemental_search_rounds": state.get("supplemental_search_round", 0),
            "source_stats": state.get("source_search_stats", {}),
            "warnings": state.get("warnings", []),
            "degraded": bool(state.get("degraded")),
            "pdf_cleanup_error": state.get("pdf_cleanup_error"),
            "remote_task_state": state.get("remote_task_state"),
            "task_status_update_error": state.get("task_status_update_error"),
        },
        artifact_refs=[],
        retryable=stage == "partial_failed",
        error_code=(None if stage in {"completed", "partial_failed"} else stage),
        error_message=state.get("error"),
    )
