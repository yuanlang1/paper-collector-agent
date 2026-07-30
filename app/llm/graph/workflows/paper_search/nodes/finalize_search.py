from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _positive_int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


async def finalize_source_search_node(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    active_sources = [
        str(source)
        for source in state.get("active_search_sources", [])
    ]
    summaries = state.get("source_summaries") or {}
    stats = state.get("source_search_stats") or {}

    active_summaries = [
        summaries.get(source)
        for source in active_sources
    ]
    failed_sources = [
        source
        for source, summary in zip(
            active_sources,
            active_summaries,
            strict=True,
        )
        if not isinstance(summary, dict)
        or summary.get("status") == "failed"
    ]
    partial_sources = [
        source
        for source, summary in zip(
            active_sources,
            active_summaries,
            strict=True,
        )
        if isinstance(summary, dict)
        and summary.get("status") == "partial"
    ]
    progress = state.get("progress") or {}
    effective_count = (
        _positive_int(progress.get("new_candidates"))
        + _positive_int(progress.get("existing_in_database"))
    )
    discovered_count = sum(
        _positive_int(
            summary.get("discovered_count")
            if isinstance(summary, dict)
            else 0
        )
        for summary in stats.values()
    )

    if active_sources and len(failed_sources) == len(active_sources):
        if effective_count == 0:
            return {
                "stage": "failed",
                "status": "failed",
                "progress": {
                    **progress,
                    "discovered": discovered_count,
                },
                "error": "所有检索来源均失败。",
            }

    degraded = bool(failed_sources or partial_sources)
    return {
        "stage": "normalizing",
        "status": "partial_failed" if degraded else "running",
        "degraded": bool(state.get("degraded")) or degraded,
        "progress": {
            **progress,
            "discovered": discovered_count,
        },
        "error": None,
    }
