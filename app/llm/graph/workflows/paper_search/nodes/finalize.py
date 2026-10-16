from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.paper_search.state import PaperSearchOutputState


logger = logging.getLogger(__name__)


async def finalize_paper_search_node(
    state: Mapping[str, Any],
) -> PaperSearchOutputState:
    try:
        await LocalArtifactStore().delete_run_directories(
            (state["child_run_id"],)
        )
    except OSError:
        logger.warning("Paper-search artifact cleanup failed.", exc_info=True)

    stage = str(state.get("stage"))
    progress = state.get("progress") or {}
    status = "success" if stage == "completed" else "timed_out" if stage == "timed_out" else "failed"

    if stage == "completed":
        summary = (
            f"论文检索完成：推荐 {progress.get('recommendation_kept', 0)} 篇，"
            f"保存关系 {progress.get('persistence_saved', 0)} 条。"
        )
    else:
        summary = "论文检索未完成。"

    lines = [summary]
    if state.get("paper_service_task_id") is not None:
        lines.append(f"检索任务 ID：{state['paper_service_task_id']}。")
    lines.append(
        f"检索 {progress.get('discovered', 0)} 篇，去重后 {progress.get('deduplicated', 0)} 篇，"
        f"已有论文 {progress.get('existing_in_database', 0)} 篇，新增候选 {progress.get('new_candidates', 0)} 篇。"
    )
    lines.append(f"补充检索轮数：{state.get('supplemental_search_round', 0)}。")
    if state.get("source_search_stats"):
        lines.append(f"来源统计：{state['source_search_stats']}")
    if state.get("remote_task_state"):
        lines.append(f"远端任务状态：{state['remote_task_state']}。")
    if state.get("error"):
        lines.append(f"失败原因：{state['error']}")
    if state.get("task_status_update_error"):
        lines.append(f"任务状态更新失败：{state['task_status_update_error']}")
    lines.extend(f"警告：{warning}" for warning in state.get("warnings", []))
    return {"content": "\n".join(lines), "status": status}
