from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.llm.graph.workflows.progress import (
    Node,
    instrument_subagent_progress_node,
)

_PHASES = {
    "initialize": ("prepare", "准备检索"),
    "plan_search": ("prepare", "理解并生成检索计划"),
    "search": ("search", "Google Scholar 检索"),
    "filter": ("normalize", "清洗检索结果"),
    "review": ("review", "审核结果并规划补充检索"),
    "enrich": ("enrich", "补充论文信息"),
    "crossref_enrich": ("enrich", "补充 Crossref 信息"),
    "venue": ("enrich", "解析期刊或会议"),
    "download_pdf": ("enrich", "下载 PDF"),
    "abstract_enrich": ("enrich", "补充摘要"),
    "recommend": ("recommend", "推荐论文"),
    "create_task": ("persist", "创建检索任务"),
    "persist": ("persist", "保存论文"),
    "finalize_pdfs": ("persist", "上传并清理临时 PDF"),
    "update_task_status": ("persist", "同步任务状态"),
    "finalize_result": ("persist", "汇总检索结果"),
}

_PROGRESS_KEYS = (
    "discovered",
    "deduplicated",
    "existing_in_database",
    "new_candidates",
    "recommendation_kept",
    "persistence_saved",
)


def _data(
    *,
    node_name: str,
    state: Mapping[str, Any],
    update: Mapping[str, Any] | None,
) -> dict[str, Any]:
    progress = {
        **dict(state.get("progress") or {}),
        **dict((update or {}).get("progress") or {}),
    }
    data: dict[str, Any] = {
        "counts": {
            key: progress[key]
            for key in _PROGRESS_KEYS
            if key in progress
        },
    }
    source_stats = (update or {}).get(
        "source_search_stats",
        state.get("source_search_stats"),
    )
    if node_name == "search" and isinstance(source_stats, Mapping):
        data["source"] = "Google Scholar"
        data["source_stats"] = source_stats.get("Google Scholar", {})
    return data


def instrument_paper_search_node(*, node_name: str, node: Node) -> Node:
    """Emit paper-search progress without creating timeline events."""

    phase, phase_label = _PHASES[node_name]
    return instrument_subagent_progress_node(
        node_name=node_name,
        node=node,
        phase=phase,
        phase_label=phase_label,
        data_builder=lambda state, update: _data(
            node_name=node_name,
            state=state,
            update=update,
        ),
        iteration_resolver=lambda state: (
            int(state.get("supplemental_search_round", 0) or 0) or None
        ),
    )
