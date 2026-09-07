import json
from datetime import date
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.model_factory import create_validated_structured_chat_model
from app.llm.subagents.paper_search import PaperSearchConstraints
from app.llm.tools.task_tools.search_task.args import (
    PromptUnderstandingArgs,
    SearchTagArgs,
)


SEARCH_TAG_SYSTEM_PROMPT = """
你负责根据用户的论文检索请求生成 search_tag。

- 仅输出 SearchTagArgs 定义的字段。 
- yearTag 必须与 query_understanding 中明确的年份范围一致；未指定年份时为 0。
- paperTag 选择用户明确需要的论文类型；若未限定，保留期刊论文和会议论文。
- sourceTag 选择适合主题与用户要求的检索来源；若未限定，使用 arXiv、DBLP 和 Google Scholar。
- 不得编造用户未提出的来源、年份或论文类型限制。
""".strip()


def derive_year_tag(
    understanding: PromptUnderstandingArgs,
) -> int:
    if understanding.yearFrom is None and understanding.yearTo is None:
        return 0

    if understanding.yearFrom is None or understanding.yearTo is None:
        raise ValueError("yearFrom 和 yearTo 必须同时提供或同时为空。")

    current_year = date.today().year

    if understanding.yearTo != current_year:
        return 0
        raise ValueError("yearTag 仅支持“最近 N 年”；指定历史区间时需扩展 search_tag 为 yearFrom/yearTo。")

    return current_year - understanding.yearFrom + 1


def with_derived_year_tag(
    search_tag: SearchTagArgs,
    understanding: PromptUnderstandingArgs,
) -> SearchTagArgs:
    return SearchTagArgs.model_validate(
        {
            **search_tag.model_dump(mode="json"),
            "yearTag": derive_year_tag(understanding),
        }
    )


class BuildSearchTagNode:
    def __init__(
        self,
        model: Any | None = None,
        *,
        skip_confirmation: bool = False,
    ) -> None:
        self.model = model or create_validated_structured_chat_model(
            SearchTagArgs,
            temperature=0,
        )
        self.skip_confirmation = skip_confirmation

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            understanding = PromptUnderstandingArgs.model_validate(
                state.get("query_understanding")
            )
            constraints = PaperSearchConstraints.model_validate(
                state.get("paper_search_constraints") or {}
            )
            fallback = SearchTagArgs(
                yearTag=derive_year_tag(understanding)
            )
        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "confirmation_required": False,
                "error": f"论文检索标签准备失败：{exc}",
            }

        try:
            original_prompt = state.get("original_prompt")
            if not isinstance(original_prompt, str) or not original_prompt.strip():
                raise ValueError("missing original_prompt")
            result = await self.model.ainvoke(
                [
                    SystemMessage(content=SEARCH_TAG_SYSTEM_PROMPT),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "original_prompt": original_prompt.strip(),
                                "query_understanding": understanding.model_dump(
                                    mode="json"
                                ),
                            },
                            ensure_ascii=False,
                        )
                    ),
                ]
            )
            proposed = (
                result
                if isinstance(result, SearchTagArgs)
                else SearchTagArgs.model_validate(result)
            )
            search_tag = proposed
            warnings = state.get("warnings", [])
        except Exception as exc:
            search_tag = fallback
            warnings = [
                *state.get("warnings", []),
                f"search tag generation failed; defaults were used: {exc}",
            ]

        if constraints.sources:
            search_tag = SearchTagArgs.model_validate(
                {
                    **search_tag.model_dump(mode="json"),
                    "sourceTag": constraints.sources,
                }
            )
        search_tag = with_derived_year_tag(search_tag, understanding)

        return {
            "stage": (
                "generating_source_queries"
                if self.skip_confirmation
                else "awaiting_confirmation"
            ),
            "status": (
                "running"
                if self.skip_confirmation
                else "waiting_confirmation"
            ),
            "confirmation_required": not self.skip_confirmation,
            "search_tag": search_tag.model_dump(mode="json"),
            "warnings": warnings,
            "error": None,
        }


BuildDefaultSearchTagNode = BuildSearchTagNode
