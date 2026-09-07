from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from app.llm.subagents.registry import SubAgentRuntime


PaperSearchSource = Literal["arXiv", "DBLP", "Google Scholar"]


class PaperSearchConstraints(BaseModel):
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    year_to: int | None = Field(default=None, ge=1900, le=2100)
    sources: list[PaperSearchSource] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_year_range(self) -> "PaperSearchConstraints":
        if (self.year_from is None) != (self.year_to is None):
            raise ValueError("year_from and year_to must be provided together")
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from must not be after year_to")
        return self


class PaperSearchDelegation(BaseModel):
    prompt: str = Field(min_length=1, max_length=2_000)
    objective: str = "检索、筛选、推荐并保存相关论文"
    constraints: PaperSearchConstraints = Field(
        default_factory=PaperSearchConstraints
    )

def build_paper_search_runtime(
    *,
    source_query_plan_model: Any,
) -> SubAgentRuntime:
    from app.llm.graph.workflows.paper_search.workflow import (
        build_paper_search_workflow,
    )
    from app.llm.subagents.registry import (
        SubAgentRuntime,
        SubAgentSpec,
        SubAgentStreamSpec,
    )
    return SubAgentRuntime(
        spec=SubAgentSpec(
            name="paper_search_agent",
            description="检索、补充检索、筛选、推荐并保存学术论文。",
            input_model=PaperSearchDelegation,
            requires_confirmation=True,
            display_name="论文检索子代理",
            confirmation_summary="将从多个来源检索、推荐并保存论文。",
        ),
        graph=build_paper_search_workflow(
            skip_confirmation=True,
            node_overrides={
                "generate_queries": _source_query_plan_node(
                    source_query_plan_model,
                )
            },
        ),
        error_code="PAPER_SEARCH_SUBGRAPH_FAILED",
        failure_summary="论文检索工作流执行失败，未完成结果保存。",
        stream=SubAgentStreamSpec(
            workflow="paper_search",
            phases=(
                ("prepare", "准备检索"),
                ("plan", "生成检索计划"),
                ("search", "多来源检索"),
                ("filter", "清洗与质量审核"),
                ("enrich", "论文信息补充"),
                ("recommend", "论文推荐"),
                ("persist", "保存与清理"),
                ("finalize", "同步任务状态"),
            ),
            node_phases={
                "initialize": "prepare",
                "intent_understanding": "prepare",
                "build_search_tag": "prepare",
                "confirm": "prepare",
                "create_task": "persist",
                "generate_queries": "plan",
                "search_arxiv": "search",
                "search_dblp": "search",
                "search_google": "search",
                "finalize_source_search": "search",
                "filter": "filter",
                "search_review": "filter",
                "supplemental_search": "filter",
                "enrich": "enrich",
                "crossref_enrich": "enrich",
                "venue": "enrich",
                "download_pdf": "enrich",
                "abstract_enrich": "enrich",
                "recommend": "recommend",
                "persist": "persist",
                "cleanup_downloaded_pdfs": "persist",
                "update_task_status": "finalize",
                "finalize_result": "finalize",
            },
            iteration_key="supplemental_search_round",
            detail_state_keys={
                "source_stats": "source_search_stats",
                "task_status_update_error": "task_status_update_error",
                "pdf_cleanup_error": "pdf_cleanup_error",
                "degraded": "degraded",
                "remote_task_state": "remote_task_state",
            },
        ),
    )


def _source_query_plan_node(model: Any):
    from app.llm.graph.workflows.paper_search.nodes.generate_queries import (
        BuildSourceQueryPlanNode,
    )
    from app.services.setting_service import (
        default_source_limits,
        source_pagination_settings,
    )
    async def generate_queries(state: Mapping[str, Any]) -> dict[str, Any]:
        source_limits = state.get("paper_search_source_limits")
        try:
            pagination_settings = source_pagination_settings(
                source_limits or default_source_limits()
            )
        except (TypeError, ValueError) as exc:
            return {
                "stage": "blocked",
                "status": "blocked",
                "error": f"论文检索来源数量配置无效：{exc}",
            }
        return await BuildSourceQueryPlanNode(
            model=model,
            pagination_settings=pagination_settings,
        )(state)

    return generate_queries
