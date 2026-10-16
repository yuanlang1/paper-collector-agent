from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.task_indexing.state import TaskIndexingOutputState


logger = logging.getLogger(__name__)


async def finalize_task_indexing_node(
    state: Mapping[str, Any],
) -> TaskIndexingOutputState:
    try:
        await LocalArtifactStore().delete_run_directories(
            (state["child_run_id"],)
        )
    except OSError:
        logger.warning("Task-indexing artifact cleanup failed.", exc_info=True)

    stage = state.get("stage")
    counts = state.get("overall_summary", {})
    ready = int(counts.get("ready", 0))
    skipped = int(counts.get("skipped", 0))
    if stage == "completed":
        summary = f"知识库索引完成：{ready} 篇论文可用于全文检索。"
        if skipped:
            summary += f"已跳过 {skipped} 篇。"
    else:
        summary = "等待知识库索引完成超时。" if stage == "timed_out" else "知识库索引未完成。"

    lines = [summary]
    if state.get("task_id") is not None:
        lines.append(f"检索任务 ID：{state['task_id']}。")
    lines.append(
        f"论文总数 {counts.get('total', 0)} 篇，可用 {ready} 篇，跳过 {skipped} 篇，"
        f"失败 {counts.get('failed', 0)} 篇。"
    )
    lines.append(
        f"索引进度：{state.get('overall_progress_percent', 0)}%；"
        f"已提交进度：{state.get('committed_progress_percent', 0)}%。"
    )
    if state.get("error"):
        lines.append(f"失败原因（{state.get('error_code') or 'TASK_INDEXING_FAILED'}）：{state['error']}")
    lines.extend(f"警告：{warning}" for warning in state.get("warnings", []))
    return {
        "content": "\n".join(lines),
        "status": "success" if stage == "completed" else "timed_out" if stage == "timed_out" else "failed",
    }
