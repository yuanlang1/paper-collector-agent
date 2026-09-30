from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.config import settings


class SupplementalSearchPlannerNode:
    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        decision = state.get("search_review_decision") or {}
        cursors = state.get("source_search_cursors") or {}
        plans_by_source = {
            plan.get("source"): plan
            for plan in state.get("source_query_plans", [])
            if isinstance(plan, dict)
        }
        action = next(
            (
                item
                for item in sorted(
                    decision.get("source_actions", []),
                    key=lambda value: value.get("priority", 10),
                )
                if item.get("source") == "Google Scholar"
            ),
            None,
        )
        cursor = cursors.get("Google Scholar", {})
        plan_template = plans_by_source.get("Google Scholar")
        if (
            not isinstance(action, dict)
            or action.get("strategy") != "next_page"
            or not cursor.get("has_next")
            or not isinstance(plan_template, dict)
        ):
            return {
                "stage": "enriching",
                "status": "partial_failed",
                "degraded": True,
                "warnings": [*state.get("warnings", []), "无可用补充检索来源，继续后续流程。"],
            }
        plan = {
            **plan_template,
            "arguments": dict(plan_template["arguments"]),
        }
        plan["arguments"]["start"] = cursor["next_offset"]
        requested_pages = min(
            int(action.get("requested_pages", 1)),
            settings.PAPER_SEARCH_MAX_EXTRA_PAGES_PER_SOURCE,
        )
        plan["arguments"]["total_limit"] = (
            int(plan["arguments"]["num"]) * requested_pages
        )
        plan["arguments"]["max_pages"] = requested_pages
        return {
            "stage": "searching",
            "status": "running",
            "active_search_sources": ["Google Scholar"],
            "active_source_query_plans": [plan],
            "supplemental_search_round": (
                int(state.get("supplemental_search_round", 0)) + 1
            ),
        }
