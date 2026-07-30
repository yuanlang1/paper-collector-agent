from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.llm.artifacts.paper_artifacts import save_paper_search_artifact
from app.llm.graph.workflows.paper_search.nodes.search import (
    SEARCH_HANDLERS,
    _error_message,
    _source_step_key,
)


class _SourceSearchNode:
    def __init__(self, source: str) -> None:
        self.source = source

    async def __call__(self, state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
        try:
            return await self._search(state, config)
        except Exception as exc:
            summaries = dict(state.get("source_summaries", {}))
            summaries[self.source] = {
                "source": self.source,
                "status": "failed",
                "discovered_count": 0,
                "artifact_ref": None,
                "error": str(exc),
            }
            return {
                "stage": "normalizing",
                "status": "partial_failed",
                "source_summaries": summaries,
                "warnings": [
                    *state.get("warnings", []),
                    f"{self.source} 检索失败：{exc}",
                ],
            }

    async def _search(
        self,
        state: Mapping[str, Any],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        active_sources = state.get("active_search_sources") or []
        if self.source not in active_sources:
            return {}
        run_id = state.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return {"stage": "failed", "status": "failed", "error": "missing run_id"}
        plan = next(
            (item for item in state.get("active_source_query_plans", [])
             if isinstance(item, dict) and item.get("source") == self.source),
            None,
        )
        if not isinstance(plan, dict):
            return {}
        handler = SEARCH_HANDLERS[self.source]
        arguments = dict(plan["arguments"])
        page_size_key = {"arXiv": "max_results", "DBLP": "h", "Google Scholar": "num"}[self.source]
        page_size = int(arguments[page_size_key])
        db = config.get("configurable", {}).get("db")
        result = await handler(arguments, db)
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
            run_id=run_id,
            step_key=_source_step_key(self.source),
            query=result.get("query") or plan["display_query"],
            search_query=result.get("search_query"),
            ok=ok,
            papers=papers,
            total_results=total_results,
            metadata={"plan_reasoning": plan["reasoning"], "plan_arguments": plan["arguments"], "pagination": pagination, "page_errors": page_errors},
        )
        summaries = dict(state.get("source_summaries", {}))
        summaries[self.source] = {
            "source": self.source,
            "status": (
                "partial" if ok and page_errors else "success" if ok else "failed"
            ),
            "discovered_count": len(papers),
            "artifact_ref": artifact.artifact_uri,
            "error": "; ".join(page_errors) or None,
        }
        cursors = dict(state.get("source_search_cursors", {}))
        cursors[self.source] = {"next_offset": pagination.get("next_offset"), "has_next": pagination.get("has_next", False), "page_size": page_size}
        stats = dict(state.get("source_search_stats", {}))
        previous = stats.get(self.source) or {}
        stats[self.source] = {
            **summaries[self.source],
            "discovered_count": (
                int(previous.get("discovered_count") or 0)
                + len(papers)
            ),
            "last_discovered_count": len(papers),
            "searched_pages": (
                int(previous.get("searched_pages") or 0)
                + int(pagination.get("pages_fetched") or 0)
            ),
            "has_next": pagination.get("has_next", False),
            "next_offset": pagination.get("next_offset"),
        }
        return {"stage": "normalizing", "status": "partial_failed" if page_errors else "running", "source_summaries": summaries, "source_search_cursors": cursors, "source_search_stats": stats, "raw_result_artifact_refs": list(dict.fromkeys([*state.get("raw_result_artifact_refs", []), artifact.artifact_uri])), "warnings": [*state.get("warnings", []), *page_errors]}


class ArxivSearchNode(_SourceSearchNode):
    def __init__(self) -> None:
        super().__init__("arXiv")


class DblpSearchNode(_SourceSearchNode):
    def __init__(self) -> None:
        super().__init__("DBLP")


class GoogleScholarSearchNode(_SourceSearchNode):
    def __init__(self) -> None:
        super().__init__("Google Scholar")
