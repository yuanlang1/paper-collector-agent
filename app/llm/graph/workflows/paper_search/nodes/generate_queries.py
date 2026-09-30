from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import settings
from app.llm.graph.workflows.paper_search_schemas import (
    PromptUnderstandingArgs,
    SearchTagArgs,
    SourceTypeValue,
)
from app.llm.provider import ChatClient, ModelOptions
from app.llm.tools.search_tools.google_scholar.search_google_scholar import (
    GoogleScholarSearchArgs,
)


SourceName = Literal["Google Scholar"]


class SourceQueryPlan(TypedDict):
    source: SourceName
    display_query: str
    reasoning: str
    arguments: dict[str, Any]
    post_filters: dict[str, Any]


class SourceQueryPlanDraft(BaseModel):
    source: SourceTypeValue
    display_query: str = Field(min_length=1, max_length=1000)
    reasoning: str = Field(min_length=1, max_length=1000)
    arguments: dict[str, Any] = Field(default_factory=dict)


class SourceQueryPlansOutput(BaseModel):
    plans: list[SourceQueryPlanDraft] = Field(min_length=1, max_length=1)


SOURCE_QUERY_PLAN_SYSTEM_PROMPT = """
你负责为论文检索工作流生成 Google Scholar 的实际检索计划。

输入包含：
1. 已由用户确认的 query_understanding；
2. 固定为 Google Scholar 的 sourceTag。

输出格式：
- 仅输出一个合法的 JSON 对象，不要输出 Markdown、代码块或额外文字。
- 顶层字段 plans：仅包含一项检索计划。
- plans[0].source：必须是 Google Scholar。
- plans[0].display_query：展示给用户看的实际检索式。
- plans[0].reasoning：简洁说明检索式如何从 query_understanding 推导得到。
- plans[0].arguments：传给 Google Scholar API 的参数对象。

最小 JSON 输出示例：
{
  "plans": [
    {
      "source": "Google Scholar",
      "display_query": "retrieval augmented generation evaluation benchmark",
      "reasoning": "使用主题和 benchmark 关键词构造相关性检索。",
      "arguments": {
        "query": "retrieval augmented generation evaluation benchmark",
        "review_only": false
      }
    }
  ]
}

严格要求：
- 只能输出 Google Scholar，不得输出其他检索来源。
- arguments 只能使用 query、year_from、year_to、review_only。
- 不得使用 searchQuery、yearFrom、yearTo、keywords、synonyms、excludeTerms 等字段。
- 充分使用 topic、keywords、synonyms、includeTerms、excludeTerms、yearFrom、yearTo 和 intent。
- 不得编造用户未确认的主题、年份或限定条件。
- display_query 用于向用户展示实际检索式。
- excludeTerms、requiresCode、paperTag 等不能被来源 API 直接表达的限制，不要伪造到 arguments 中；它们将由后续过滤节点处理。
- Pagination fields (start, num, total_limit, max_pages) are injected by Settings and must not be output.
""".strip()


def _build_post_filters(
    understanding: PromptUnderstandingArgs,
    search_tag: SearchTagArgs,
) -> dict[str, Any]:
    return {
        "include_terms": understanding.includeTerms,
        "exclude_terms": understanding.excludeTerms,
        "requires_code": understanding.requiresCode,
        "paper_tags": [paper_type.value for paper_type in search_tag.paperTag],
    }


def _normalize_source_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "searchQuery": "query",
        "yearFrom": "year_from",
        "yearTo": "year_to",
    }
    normalized = dict(arguments)

    for alias, field_name in aliases.items():
        if field_name not in normalized and alias in normalized:
            normalized[field_name] = normalized[alias]

    return {
        key: value
        for key, value in normalized.items()
        if key in GoogleScholarSearchArgs.model_fields
    }


class BuildSourceQueryPlanNode:
    def __init__(
        self,
        model: Any | None = None,
        chat: ChatClient | None = None,
        pagination_settings: dict[str, int] | None = None,
    ) -> None:
        self.model = model or (chat or ChatClient()).structured(
            SourceQueryPlansOutput,
            options=ModelOptions(temperature=0),
        )
        self.pagination_settings = pagination_settings or {
            "google_max_pages": settings.PAPER_SEARCH_GOOGLE_SCHOLAR_MAX_PAGES,
            "google_page_size": settings.PAPER_SEARCH_GOOGLE_SCHOLAR_PAGE_SIZE,
            "google_total_limit": settings.PAPER_SEARCH_GOOGLE_SCHOLAR_TOTAL_LIMIT,
        }

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            understanding = PromptUnderstandingArgs.model_validate(
                state.get("query_understanding"),
            )
            search_tag = SearchTagArgs.model_validate(state.get("search_tag"))
            selected_source = SourceTypeValue.GOOGLE_SCHOLAR
            result = await self.model.ainvoke(
                [
                    SystemMessage(content=SOURCE_QUERY_PLAN_SYSTEM_PROMPT),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "query_understanding": understanding.model_dump(
                                    mode="json",
                                ),
                                "source_tag": [selected_source.value],
                            },
                            ensure_ascii=False,
                        ),
                    ),
                ],
            )
            output = (
                result
                if isinstance(result, SourceQueryPlansOutput)
                else SourceQueryPlansOutput.model_validate(result)
            )
            draft = output.plans[0]

            if draft.source != selected_source:
                raise ValueError("模型生成的检索来源必须是 Google Scholar。")

            arguments = _normalize_source_arguments(draft.arguments)
            arguments.update(
                start=0,
                num=min(self.pagination_settings["google_page_size"], 20),
                total_limit=self.pagination_settings["google_total_limit"],
                max_pages=self.pagination_settings["google_max_pages"],
            )
            validated_arguments = GoogleScholarSearchArgs.model_validate(arguments)
            plan: SourceQueryPlan = {
                "source": selected_source.value,
                "display_query": draft.display_query,
                "reasoning": draft.reasoning,
                "arguments": validated_arguments.model_dump(mode="json"),
                "post_filters": _build_post_filters(understanding, search_tag),
            }
        except Exception as exc:
            return {
                "stage": "blocked",
                "status": "blocked",
                "source_query_plans": [],
                "error": f"生成检索计划失败：{exc}",
            }

        return {
            "stage": "searching",
            "status": "running",
            "source_query_plans": [plan],
            "active_source_query_plans": [plan],
            "active_search_sources": [selected_source.value],
            "requested_sources": [selected_source.value],
            "supplemental_search_round": 0,
            "error": None,
        }
