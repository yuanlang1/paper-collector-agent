from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.graph.workflows.paper_search.state import SourceQueryPlan
from app.llm.graph.workflows.paper_search_schemas import (
    PromptUnderstandingArgs,
    SearchPlanningArgs,
    SearchTagArgs,
    SourceTypeValue,
)
from app.llm.provider import ChatClient, ModelOptions
from app.llm.tools.search_tools.google_scholar.search_google_scholar import (
    GoogleScholarSearchArgs,
)
from app.services.setting_service import (
    default_source_limits,
    source_pagination_settings,
)


PLAN_SEARCH_SYSTEM_PROMPT = """
你是论文检索规划专家。将用户的自然语言请求一次性转换成结构化检索意图、论文类型和唯一的 Google Scholar 查询草案。

角色要求：
- 准确提炼用户实际提出的主题、意图、年份和限制，不编造限制或事实。
- 查询来源固定为 Google Scholar；不得输出或选择其他来源。

字段规范：
- 只输出 SearchPlanningArgs 定义的合法 JSON 对象，不要输出 Markdown、代码块或额外说明。
- query_understanding.topic 使用简洁、准确的英文学术表达；除 reasoning 外，其文本字段均使用英文学术词。
- 仅当用户明确给出起止年份时同时填写 yearFrom 和 yearTo；“recent”“latest”“近年来”等相对表达必须同时为 null，不得推断年份。
- keywords 是核心检索词；synonyms 只使用可靠同义词或常见学术表达；includeTerms、excludeTerms 只保留用户明确提出的限制。
- requiresCode 仅在用户明确要求代码、实现或开源仓库时为 true，未提及时为 null。
- intent 只能是 survey、benchmark、method 或 mixed；reasoning 用简洁中文说明字段来源。
- paperTag 只选择用户明确需要的论文类型；未限定时使用 [1, 2]（期刊论文、会议论文）。
- query_plan 只包含 display_query、reasoning、arguments。arguments 只能使用 query、year_from、year_to、review_only；不得输出分页字段或来源字段。分页和来源均由代码固定。
- display_query 展示实际检索式；不能直接表达的 includeTerms、excludeTerms、requiresCode、paperTag 限制由后续过滤处理，不得伪造为 API 参数。

输出样例：
{
  "query_understanding": {
    "topic": "retrieval augmented generation",
    "subfields": [],
    "intent": "method",
    "yearFrom": null,
    "yearTo": null,
    "keywords": ["retrieval augmented generation"],
    "synonyms": ["RAG"],
    "includeTerms": [],
    "excludeTerms": [],
    "requiresCode": null,
    "reasoning": "用户请求检索 RAG 方法论文，未限定年份、代码或其他条件。"
  },
  "paperTag": [1, 2],
  "query_plan": {
    "display_query": "retrieval augmented generation method",
    "reasoning": "使用主题和方法意图构造 Google Scholar 检索式。",
    "arguments": {
      "query": "retrieval augmented generation method",
      "review_only": false
    }
  }
}
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


class PlanSearchNode:
    def __init__(
        self,
        model: Any | None = None,
        chat: ChatClient | None = None,
        pagination_settings: dict[str, int] | None = None,
    ) -> None:
        client = chat or ChatClient()
        self.model = model or client.structured(
            SearchPlanningArgs,
            options=ModelOptions(temperature=0),
        )
        self.pagination_settings = pagination_settings

    @staticmethod
    def _derive_year_tag(understanding: PromptUnderstandingArgs) -> int:
        if understanding.yearFrom is None and understanding.yearTo is None:
            return 0
        if understanding.yearFrom is None or understanding.yearTo is None:
            raise ValueError("yearFrom 和 yearTo 必须同时提供或同时为空。")

        current_year = date.today().year
        if understanding.yearTo != current_year:
            return 0
        return current_year - understanding.yearFrom + 1

    def _pagination_settings(self, state: Mapping[str, Any]) -> dict[str, int]:
        if self.pagination_settings is not None:
            return self.pagination_settings
        source_limits = state.get("paper_search_source_limits")
        return source_pagination_settings(
            source_limits or default_source_limits(),
        )

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        try:
            original_prompt = state.get("original_prompt")
            if not isinstance(original_prompt, str) or not original_prompt.strip():
                raise ValueError("paper search intent requires a non-empty original_prompt")

            result = await self.model.ainvoke(
                [
                    SystemMessage(content=PLAN_SEARCH_SYSTEM_PROMPT),
                    HumanMessage(content=original_prompt.strip()),
                ]
            )
            output = (
                result
                if isinstance(result, SearchPlanningArgs)
                else SearchPlanningArgs.model_validate(result)
            )
            understanding = output.query_understanding
            selected_source = SourceTypeValue.GOOGLE_SCHOLAR
            search_tag = SearchTagArgs(
                yearTag=self._derive_year_tag(understanding),
                paperTag=output.paperTag,
                sourceTag=[selected_source],
            )
            pagination_settings = self._pagination_settings(state)
            arguments = _normalize_source_arguments(output.query_plan.arguments)
            arguments.update(
                year_from=understanding.yearFrom or 2000,
                year_to=understanding.yearTo or date.today().year,
                start=0,
                num=min(pagination_settings["google_page_size"], 20),
                total_limit=pagination_settings["google_total_limit"],
                max_pages=pagination_settings["google_max_pages"],
            )
            validated_arguments = GoogleScholarSearchArgs.model_validate(arguments)
            plan: SourceQueryPlan = {
                "source": selected_source.value,
                "display_query": output.query_plan.display_query,
                "reasoning": output.query_plan.reasoning,
                "arguments": validated_arguments.model_dump(mode="json"),
                "post_filters": _build_post_filters(understanding, search_tag),
            }
        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "query_understanding": None,
                "search_tag": None,
                "source_query_plans": [],
                "active_source_query_plans": [],
                "active_search_sources": [],
                "requested_sources": [],
                "intent_error": str(exc),
                "error": f"论文检索计划失败：{exc}",
            }

        return {
            "stage": "searching",
            "status": "running",
            "query_understanding": understanding.model_dump(mode="json"),
            "search_tag": search_tag.model_dump(mode="json"),
            "source_query_plans": [plan],
            "active_source_query_plans": [plan],
            "active_search_sources": [selected_source.value],
            "requested_sources": [selected_source.value],
            "supplemental_search_round": 0,
            "warnings": list(state.get("warnings", [])),
            "intent_error": None,
            "error": None,
        }
