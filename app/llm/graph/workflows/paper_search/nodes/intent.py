from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.model_factory import create_validated_structured_chat_model
from app.llm.subagents.paper_search.contracts import PaperSearchConstraints
from app.llm.tools.task_tools.search_task.args import PromptUnderstandingArgs


PAPER_SEARCH_INTENT_SYSTEM_PROMPT = """
    你负责把用户的论文检索请求转换成结构化检索意图。

    严格遵守：
    - 只输出 PromptUnderstandingArgs 定义的字段。
    - topic 应简洁、准确地概括研究主题。
    - 仅在用户明确给出起止年份时填写 yearFrom 和 yearTo；未知时两者必须同时为 null。“recent”“latest”“近年来”等未给出具体年份的相对时间表达不属于明确年份范围，两个字段必须为 null，不得自行推断年份。
    - keywords 是核心检索词；synonyms 仅使用可靠的同义词或常用学术表达。
    - includeTerms 和 excludeTerms 仅保留用户明确提出的限制。
    - requiresCode 仅在用户明确要求代码、实现或开源仓库时为 true；未提及时为 null。
    - intent 只能是 survey、benchmark、method 或 mixed。
    - reasoning 使用简洁中文说明字段如何从用户请求中得出，不得虚构限制或事实。
    - 除了reasoning，其他字段要以英文学术词给出，因为后面要
""".strip()


class IntentUnderstandingNode:

    def __init__(
        self,
         model: Any | None = None
    ) -> None:
        self.model = model or create_validated_structured_chat_model(
            PromptUnderstandingArgs,
            temperature=0,
        )

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        try:
            original_prompt = state.get("original_prompt")

            if not isinstance(original_prompt, str) or not original_prompt.strip():
                raise ValueError(
                    "paper search intent requires a non-empty original_prompt"
                )

            result = await self.model.ainvoke(
                [
                    SystemMessage(content=PAPER_SEARCH_INTENT_SYSTEM_PROMPT),
                    HumanMessage(content=original_prompt.strip()),
                ]
            )

            understanding = (
                result
                if isinstance(result, PromptUnderstandingArgs)
                else PromptUnderstandingArgs.model_validate(result)
            )
            constraints = PaperSearchConstraints.model_validate(
                state.get("paper_search_constraints") or {}
            )
            if constraints.year_from is not None:
                understanding = understanding.model_copy(
                    update={
                        "yearFrom": constraints.year_from,
                        "yearTo": constraints.year_to,
                    }
                )
        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "intent_error": str(exc),
                "error": f"论文检索意图理解失败：{exc}",
            }

        return {
            "stage": "building_search_tag",
            "status": "running",
            "query_understanding": understanding.model_dump(mode="json"),
            "intent_error": None,
            "error": None,
        }
