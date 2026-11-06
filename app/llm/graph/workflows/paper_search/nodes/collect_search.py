from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class CollectSearchNode:
    """Collect the current parallel batch and rebuild cumulative search statistics."""

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        try:
            round_number = state["supplemental_search_round"]
            results = state.get("search_results", {})
            batch = {}
            for provider in state["active_search_tasks"]:
                result = results.get(provider)
                if result is None or result["round"] != round_number:
                    raise ValueError(f"Missing current search result: {provider}")
                batch[provider] = result
            history = {**state.get("search_history", {}), str(round_number): batch}
            stats, cursors, refs, warnings = {}, {}, [], []
            for number in sorted(history, key=int):
                for provider, result in history[number].items():
                    previous = stats.get(provider, {})
                    stats[provider] = {
                        "status": result["status"],
                        "discovered_count": previous.get("discovered_count", 0)
                        + result["returned_count"],
                        "searched_pages": previous.get("searched_pages", 0)
                        + result["pages_fetched"],
                        "has_next": result["next_cursor"] is not None,
                    }
                    cursors[provider] = result["next_cursor"]
                    if result["artifact_ref"]:
                        refs.append(result["artifact_ref"])
                    warnings.extend(result["warnings"])
                    if result["error"]:
                        warnings.append(f"{provider}: {result['error']}")
            discovered = sum(item["discovered_count"] for item in stats.values())
            progress = state.get("progress", {})
            effective = progress.get("new_candidates", 0) + progress.get(
                "existing_in_database", 0
            )
            failed = (
                bool(batch)
                and all(item["status"] == "failed" for item in batch.values())
                and not effective
            )
            return {
                "search_history": history,
                "provider_stats": stats,
                "provider_cursors": cursors,
                "raw_result_artifact_refs": list(dict.fromkeys(refs)),
                "warnings": list(dict.fromkeys([*state.get("warnings", []), *warnings])),
                "progress": {**state.get("progress", {}), "discovered": discovered},
                "stage": "failed" if failed else "normalizing",
                "status": "failed" if failed else "running",
                "error": "所有检索来源均失败。" if failed else None,
            }
        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"检索结果汇总失败：{exc}",
            }
