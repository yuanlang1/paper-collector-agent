from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import settings
from app.llm.model_factory import create_structured_chat_model
from app.llm.tools.search_tools.arxiv.search_args import (
    ArxivSearchArgs,
)
from app.llm.tools.search_tools.dblp.search_args import (
    DblpSearchArgs,
)
from app.llm.tools.search_tools.crossref.search_args import (
    CrossrefSearchArgs,
)
from app.llm.tools.search_tools.google_scholar.search_args import (
    GoogleScholarSearchArgs,
)
from app.llm.tools.task_tools.search_task.args import (
    PromptUnderstandingArgs,
    SearchTagArgs,
    SourceTypeValue,
)


SourceName = Literal["arXiv", "DBLP", "Crossref", "Google Scholar"]


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
    plans: list[SourceQueryPlanDraft] = Field(min_length=1)


SOURCE_ARGUMENT_MODELS = {
    SourceTypeValue.ARXIV: ArxivSearchArgs,
    SourceTypeValue.DBLP: DblpSearchArgs,
    SourceTypeValue.CROSSREF: CrossrefSearchArgs,
    SourceTypeValue.GOOGLE_SCHOLAR: GoogleScholarSearchArgs,
}

SOURCE_ARGUMENT_ALIASES = {
    "searchQuery": "query",
    "yearFrom": "year_from",
    "yearTo": "year_to",
}

ARXIV_SORT_ALIASES = {
    "submittedDate": "newest",
    "lastUpdatedDate": "updated",
}

ARXIV_SEARCH_TYPE_ALIASES = {
    "all": "topic",
}


SOURCE_QUERY_PLAN_SYSTEM_PROMPT = """
    你负责为论文检索工作流生成各检索来源的实际检索计划。

    输入包含：
    1. 已由用户确认的 query_understanding；
    2. 前端选择的 sourceTag。

    严格要求：
    - 只能为 sourceTag 中的来源生成计划，且每个来源必须恰好一条。
    - 只能输出 arXiv、DBLP、Crossref、Google Scholar 四种来源。
    - arguments 必须严格符合对应来源的参数模型：
    - arXiv: ArxivSearchArgs
    - DBLP: DblpSearchArgs
    - Crossref: CrossrefSearchArgs
    - Google Scholar: GoogleScholarSearchArgs
    - arguments 中只能使用以下 snake_case 字段：
    - arXiv: query, arxiv_ids, search_type, category, year_from, year_to, start, max_results, total_limit, sort, include_abstract。
      search_type 只能是 topic、title、author、abstract 或 category；广义检索使用 topic，不得输出 all。
      其中 sort 只能是 relevance、newest、updated，不能使用 submittedDate 或 lastUpdatedDate。
    - DBLP: query, f, h, total_limit, year_from, year_to, venue, paper_type
    - Crossref: doi, query, title, author, issn, from_pub_date,
      until_pub_date, work_type, limit
    - Google Scholar: query, start, num, total_limit, year_from, year_to, review_only
    - 不得使用 searchQuery、yearFrom、yearTo、keywords、synonyms、excludeTerms 等字段。
    - 充分使用 topic、keywords、synonyms、includeTerms、excludeTerms、
    yearFrom、yearTo 和 intent。
    - 不得编造用户未确认的主题、年份、限定条件或来源。
    - display_query 用于向用户展示实际检索式。
    - reasoning 简洁说明该来源的检索式如何从 query_understanding 推导得出。
    - excludeTerms、requiresCode、paperTag 等不能被来源 API 直接表达的限制，
    不要伪造到 arguments 中；它们将由后续过滤节点处理。
    - Pagination fields (start, f, max_results, h, num, total_limit, max_pages) are injected by Settings and must not be output.
""".strip()


def _build_post_filters(
    understanding: PromptUnderstandingArgs,
    search_tag: SearchTagArgs,
) -> dict[str, Any]:
    return {
        "include_terms": understanding.includeTerms,
        "exclude_terms": understanding.excludeTerms,
        "requires_code": understanding.requiresCode,
        "paper_tags": [
            paper_type.value
            for paper_type in search_tag.paperTag
        ],
    }


def _normalize_source_arguments(
    arguments: dict[str, Any],
    args_model: type[BaseModel],
) -> dict[str, Any]:
    normalized = dict(arguments)

    for alias, field_name in SOURCE_ARGUMENT_ALIASES.items():
        if field_name not in normalized and alias in normalized:
            normalized[field_name] = normalized[alias]

    if args_model is ArxivSearchArgs:
        sort = normalized.get("sort")
        if isinstance(sort, str):
            normalized["sort"] = ARXIV_SORT_ALIASES.get(
                sort,
                sort,
            )

        search_type = normalized.get("search_type")
        if isinstance(search_type, str):
            normalized["search_type"] = ARXIV_SEARCH_TYPE_ALIASES.get(
                search_type,
                search_type,
            )

    return {
        key: value
        for key, value in normalized.items()
        if key in args_model.model_fields
    }


class BuildSourceQueryPlanNode:
    def __init__(
        self,
        model: Any | None = None,
        pagination_settings: dict[str, int] | None = None,
    ) -> None:
        self.model = model or create_structured_chat_model(
            SourceQueryPlansOutput,
            temperature=0,
        )
        self.pagination_settings = pagination_settings or {
            "arxiv_max_pages": settings.PAPER_SEARCH_ARXIV_MAX_PAGES,
            "arxiv_page_size": settings.PAPER_SEARCH_ARXIV_PAGE_SIZE,
            "arxiv_total_limit": settings.PAPER_SEARCH_ARXIV_TOTAL_LIMIT,
            "dblp_max_pages": settings.PAPER_SEARCH_DBLP_MAX_PAGES,
            "dblp_page_size": settings.PAPER_SEARCH_DBLP_PAGE_SIZE,
            "dblp_total_limit": settings.PAPER_SEARCH_DBLP_TOTAL_LIMIT,
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
            search_tag = SearchTagArgs.model_validate(
                state.get("search_tag"),
            )

            selected_sources = [
                source
                for source in search_tag.sourceTag
                if source != SourceTypeValue.CROSSREF
            ]

            model_input = {
                "query_understanding": understanding.model_dump(mode="json"),
                "source_tag": [source.value for source in selected_sources],
            }

            result = await self.model.ainvoke(
                [
                    SystemMessage(content = SOURCE_QUERY_PLAN_SYSTEM_PROMPT),
                    HumanMessage(
                        content=json.dumps(model_input, ensure_ascii=False),
                    ),
                ]
            )

            output = (
                result
                if isinstance(result, SourceQueryPlansOutput)
                else SourceQueryPlansOutput.model_validate(result)
            )

            returned_sources = [plan.source for plan in output.plans]

            if len(returned_sources) != len(set(returned_sources)):
                raise ValueError("同一检索来源不能生成多个检索计划。")

            if set(returned_sources) != set(selected_sources):
                raise ValueError("模型生成的检索来源必须与前端 sourceTag 完全一致。")

            post_filters = _build_post_filters(understanding, search_tag,)
            plans: list[SourceQueryPlan] = []

            for draft in output.plans:
                args_model = SOURCE_ARGUMENT_MODELS[draft.source]
                arguments = _normalize_source_arguments(
                    draft.arguments,
                    args_model,
                )

                if draft.source == SourceTypeValue.ARXIV:
                    arguments.update(
                        start=0,
                        max_results=min(self.pagination_settings["arxiv_page_size"], 2000),
                        total_limit=min(self.pagination_settings["arxiv_total_limit"], 30_000),
                        max_pages=self.pagination_settings["arxiv_max_pages"],
                    )
                elif draft.source == SourceTypeValue.DBLP:
                    arguments.update(
                        f=0,
                        h=min(self.pagination_settings["dblp_page_size"], 1000),
                        total_limit=self.pagination_settings["dblp_total_limit"],
                        max_pages=self.pagination_settings["dblp_max_pages"],
                    )
                elif draft.source == SourceTypeValue.GOOGLE_SCHOLAR:
                    arguments.update(
                        start=0,
                        num=min(self.pagination_settings["google_page_size"], 20),
                        total_limit=self.pagination_settings["google_total_limit"],
                        max_pages=self.pagination_settings["google_max_pages"],
                    )

                validated_arguments = args_model.model_validate(arguments)

                plans.append(
                    {
                        "source": draft.source.value,
                        "display_query": draft.display_query,
                        "reasoning": draft.reasoning,
                        "arguments": validated_arguments.model_dump(mode="json"),
                        "post_filters": post_filters,
                    }
                )

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
            "source_query_plans": plans,
            "active_source_query_plans": plans,
            "active_search_sources": [source.value for source in selected_sources],
            "requested_sources": [
                source.value for source in selected_sources
            ],
            "supplemental_search_round": 0,
            "error": None,
        }
