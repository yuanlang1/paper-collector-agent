from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig
from sqlalchemy.orm import Session

from app.llm.artifacts.paper_artifacts import save_paper_search_artifact
from app.llm.tools.search_tools.google_scholar.search_google_scholar import (
    google_scholar_search_service,
)


SearchHandler = Callable[
    [dict[str, Any], Session | None],
    Awaitable[dict[str, Any]],
]

SEARCH_HANDLERS: dict[str, SearchHandler] = {
    "Google Scholar": google_scholar_search_service,
}


def _source_step_key(source: str) -> str:
    return "search_" + source.lower().replace(" ", "_").replace("-", "_")


def _error_message(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        message = value.get("message") or value.get("code")
        return str(message) if message else str(value)
    return str(value)


def _positive_int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


class SearchNode:
    source = "Google Scholar"

    async def __call__(
        self,
        state: Mapping[str, Any],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        child_run_id = state["child_run_id"]

        try:
            update = await self._search(state, config, child_run_id)
        except Exception as exc:
            summaries = dict(state.get("source_summaries", {}))
            summaries[self.source] = {
                "source": self.source,
                "status": "failed",
                "discovered_count": 0,
                "artifact_ref": None,
                "error": str(exc),
            }
            update = {
                "source_summaries": summaries,
                "warnings": [
                    *state.get("warnings", []),
                    f"{self.source} 检索失败：{exc}",
                ],
            }

        return self._finalize(state, update)

    async def _search(
        self,
        state: Mapping[str, Any],
        config: RunnableConfig,
        child_run_id: str,
    ) -> dict[str, Any]:
        del config
        active_sources = state.get("active_search_sources") or []
        if self.source not in active_sources:
            return {}

        plan = next(
            (
                item
                for item in state.get("active_source_query_plans", [])
                if isinstance(item, dict) and item.get("source") == self.source
            ),
            None,
        )
        if not isinstance(plan, dict):
            return {}

        arguments = dict(plan["arguments"])
        page_size = int(arguments["num"])
        result = await SEARCH_HANDLERS[self.source](arguments, None)
        metadata = result.get("metadata") or {}
        pagination = dict(metadata.get("pagination") or {})
        papers = [
            item
            for item in result.get("papers") or []
            if isinstance(item, dict)
        ]
        page_errors = list(result.get("warnings") or [])
        if result.get("ok") is not True:
            page_errors.append(_error_message(result.get("error")) or "search failed")
        ok = result.get("ok") is True
        try:
            total_results = int(metadata.get("total_results") or 0)
        except (TypeError, ValueError):
            total_results = 0

        artifact = await save_paper_search_artifact(
            source=self.source,
            run_id=child_run_id,
            step_key=_source_step_key(self.source),
            query=result.get("query") or plan["display_query"],
            search_query=result.get("search_query"),
            ok=ok,
            papers=papers,
            total_results=total_results,
            metadata={
                "plan_reasoning": plan["reasoning"],
                "plan_arguments": plan["arguments"],
                "pagination": pagination,
                "page_errors": page_errors,
            },
        )
        summaries = dict(state.get("source_summaries", {}))
        summaries[self.source] = {
            "source": self.source,
            "status": "partial" if ok and page_errors else "success" if ok else "failed",
            "discovered_count": len(papers),
            "artifact_ref": artifact.artifact_uri,
            "error": "; ".join(page_errors) or None,
        }
        cursors = dict(state.get("source_search_cursors", {}))
        cursors[self.source] = {
            "next_offset": pagination.get("next_offset"),
            "has_next": pagination.get("has_next", False),
            "page_size": page_size,
        }
        stats = dict(state.get("source_search_stats", {}))
        previous = stats.get(self.source) or {}
        stats[self.source] = {
            **summaries[self.source],
            "discovered_count": (
                int(previous.get("discovered_count") or 0) + len(papers)
            ),
            "last_discovered_count": len(papers),
            "searched_pages": (
                int(previous.get("searched_pages") or 0)
                + int(pagination.get("pages_fetched") or 0)
            ),
            "has_next": pagination.get("has_next", False),
            "next_offset": pagination.get("next_offset"),
        }
        return {
            "source_summaries": summaries,
            "source_search_cursors": cursors,
            "source_search_stats": stats,
            "raw_result_artifact_refs": list(
                dict.fromkeys([
                    *state.get("raw_result_artifact_refs", []),
                    artifact.artifact_uri,
                ])
            ),
            "warnings": [*state.get("warnings", []), *page_errors],
        }

    def _finalize(
        self,
        state: Mapping[str, Any],
        update: dict[str, Any],
    ) -> dict[str, Any]:
        active_sources = [
            str(source) for source in state.get("active_search_sources", [])
        ]
        summaries = update.get("source_summaries", state.get("source_summaries", {}))
        stats = update.get("source_search_stats", state.get("source_search_stats", {}))
        progress = {
            **(state.get("progress") or {}),
            **(update.get("progress") or {}),
        }
        failed_sources = [
            source
            for source in active_sources
            if not isinstance(summaries.get(source), dict)
            or summaries[source].get("status") == "failed"
        ]
        partial_sources = [
            source
            for source in active_sources
            if isinstance(summaries.get(source), dict)
            and summaries[source].get("status") == "partial"
        ]
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
        progress["discovered"] = discovered_count

        if active_sources and len(failed_sources) == len(active_sources):
            if effective_count == 0:
                return {
                    **update,
                    "stage": "failed",
                    "status": "failed",
                    "progress": progress,
                    "error": "所有检索来源均失败。",
                }

        degraded = bool(failed_sources or partial_sources)
        return {
            **update,
            "stage": "normalizing",
            "status": "partial_failed" if partial_sources else "running",
            "degraded": bool(state.get("degraded")) or degraded,
            "progress": progress,
            "error": None,
        }
