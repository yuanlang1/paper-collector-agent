from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.config import settings


class SupplementalSearchPlannerNode:
    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        decision = state.get("search_review_decision") or {}
        cursors = state.get("source_search_cursors") or {}
        plans_by_source = {plan.get("source"): plan for plan in state.get("source_query_plans", []) if isinstance(plan, dict)}
        active_plans: list[dict[str, Any]] = []
        for action in sorted(decision.get("source_actions", []), key=lambda value: value.get("priority", 10)):
            source = action.get("source")
            cursor = cursors.get(source, {})
            if action.get("strategy") != "next_page" or not cursor.get("has_next") or source not in plans_by_source:
                continue
            plan = {**plans_by_source[source], "arguments": dict(plans_by_source[source]["arguments"])}
            offset_key = "f" if source == "DBLP" else "start"
            plan["arguments"][offset_key] = cursor["next_offset"]
            page_size_key = {
                "arXiv": "max_results",
                "DBLP": "h",
                "Google Scholar": "num",
            }[source]
            requested_pages = min(
                int(action.get("requested_pages", 1)),
                settings.PAPER_SEARCH_MAX_EXTRA_PAGES_PER_SOURCE,
            )
            plan["arguments"]["total_limit"] = (
                int(plan["arguments"][page_size_key]) * requested_pages
            )
            plan["arguments"]["max_pages"] = requested_pages
            active_plans.append(plan)
        if not active_plans:
            return {"stage": "enriching", "status": "partial_failed", "degraded": True, "warnings": [*state.get("warnings", []), "无可用补充检索来源，继续后续流程。"]}
        return {"stage": "searching", "status": "running", "active_search_sources": [plan["source"] for plan in active_plans], "active_source_query_plans": active_plans, "supplemental_search_round": int(state.get("supplemental_search_round", 0)) + 1}
