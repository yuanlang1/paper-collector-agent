from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from sqlalchemy.orm import Session

from app.llm.artifacts.paper_artifacts import (
    save_paper_search_artifact,
)
from app.llm.tools.search_tools.arxiv.search_arxiv import (
    arxiv_search_handler,
)
from app.llm.tools.search_tools.dblp.search_dblp import (
    dblp_search_handler,
)
from app.llm.tools.search_tools.crossref.search_crossref import (
    crossref_search_handler,
)
from app.llm.tools.search_tools.google_scholar.search_google_scholar import (
    google_scholar_search_handler,
)


class SourceQueryPlan(TypedDict):
    source: str
    display_query: str
    reasoning: str
    arguments: dict[str, Any]
    post_filters: dict[str, Any]


SearchHandler = Callable[
    [dict[str, Any], Session | None],
    Awaitable[dict[str, Any]],
]

SEARCH_HANDLERS: dict[str, SearchHandler] = {
    "arXiv": arxiv_search_handler,
    "DBLP": dblp_search_handler,
    "Crossref": crossref_search_handler,
    "Google Scholar": google_scholar_search_handler,
}


def _source_step_key(source: str) -> str:
    return (
        "search_"
        + source.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _error_message(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, dict):
        message = value.get("message") or value.get("code")
        return str(message) if message else str(value)

    return str(value)


class MultiSourceSearchNode:
    async def __call__(
        self,
        state: Mapping[str, Any],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        query_plans = state.get("source_query_plans") or []
        run_id = state.get("run_id")

        if not isinstance(run_id, str) or not run_id:
            return {
                "stage": "failed",
                "status": "failed",
                "error": "检索任务缺少有效 run_id。",
            }

        if not isinstance(query_plans, list) or not query_plans:
            return {
                "stage": "blocked",
                "status": "blocked",
                "error": "检索任务缺少 source_query_plans。",
            }

        async def search_one(
            plan: SourceQueryPlan,
        ) -> dict[str, Any]:
            source = plan.get("source")
            handler = SEARCH_HANDLERS.get(source)

            if handler is None:
                return {
                    "source": source or "unknown",
                    "status": "failed",
                    "discovered_count": 0,
                    "artifact_ref": None,
                    "error": f"不支持的检索来源：{source}",
                }

            try:
                raw_result = await handler(
                    plan["arguments"],
                    None,
                )

                ok = bool(raw_result.get("ok"))
                papers = raw_result.get("papers") or []

                if not isinstance(papers, list):
                    papers = []

                metadata = raw_result.get("metadata") or {}
                total_results = int(
                    metadata.get("total_results") or 0,
                )

                artifact = await save_paper_search_artifact(
                    source=source,
                    run_id=run_id,
                    step_key=_source_step_key(source),
                    query=raw_result.get("query") or plan["display_query"],
                    search_query=raw_result.get("search_query"),
                    ok=ok,
                    papers=papers,
                    total_results=total_results,
                    metadata={
                        "plan_reasoning": plan["reasoning"],
                        "plan_arguments": plan["arguments"],
                        "post_filters": plan["post_filters"],
                        "source_metadata": metadata,
                        "source_error": raw_result.get("error"),
                    },
                )

                return {
                    "source": source,
                    "status": "success" if ok else "failed",
                    "discovered_count": len(papers),
                    "artifact_ref": artifact.artifact_uri,
                    "error": (
                        None
                        if ok
                        else _error_message(raw_result.get("error"))
                    ),
                }

            except Exception as exc:
                return {
                    "source": source,
                    "status": "failed",
                    "discovered_count": 0,
                    "artifact_ref": None,
                    "error": str(exc),
                }

        summaries = await asyncio.gather(
            *[
                search_one(plan)
                for plan in query_plans if isinstance(plan, dict)
            ]
        )

        source_summaries = {
            summary["source"]: summary
            for summary in summaries
        }
        artifact_refs = [
            summary["artifact_ref"]
            for summary in summaries if summary["artifact_ref"]
        ]
        failed_summaries = [
            summary
            for summary in summaries
            if summary["status"] == "failed"
        ]
        discovered_count = sum(
            summary["discovered_count"]
            for summary in summaries
        )

        if len(failed_summaries) == len(summaries):
            return {
                "stage": "failed",
                "status": "failed",
                "source_summaries": source_summaries,
                "raw_result_artifact_refs": artifact_refs,
                "warnings": [
                    *state.get("warnings", []),
                    *[
                        f"{summary['source']} 检索失败："
                        f"{summary['error']}"
                        for summary in failed_summaries
                    ],
                ],
                "error": "所有检索来源均失败。",
            }

        return {
            "stage": "normalizing",
            "status": (
                "partial_failed"
                if failed_summaries
                else "running"
            ),
            "source_summaries": source_summaries,
            "raw_result_artifact_refs": list(
                dict.fromkeys(
                    [
                        *state.get(
                            "raw_result_artifact_refs",
                            [],
                        ),
                        *artifact_refs,
                    ]
                )
            ),
            "progress": {
                **state.get("progress", {}),
                "searched_sources": len(summaries),
                "discovered": discovered_count,
            },
            "warnings": [
                *state.get("warnings", []),
                *[
                    f"{summary['source']} 检索失败："
                    f"{summary['error']}"
                    for summary in failed_summaries
                ],
            ],
            "error": None,
        }
