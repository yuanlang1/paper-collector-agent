from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import set_config_context
from langgraph.types import interrupt

from app.llm.graph.workflows.paper_search.nodes.build_search_tag import with_derived_year_tag
from app.llm.tools.task_tools.search_task.args import (
    PromptUnderstandingArgs,
    SearchTagArgs,
)


def _interrupt_with_config(
    config: RunnableConfig,
    payload: dict[str, Any],
) -> Any:
    with set_config_context(config) as context:
        return context.run(interrupt, payload)


def _build_confirmation_payload(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "type": "paper_search_intent_confirmation",
        "local_task_id": state.get("local_task_id"),
        "original_prompt": state.get("original_prompt"),
        "query_understanding": state.get("query_understanding"),
        "search_tag": state.get("search_tag"),
        "message": "请确认或修改以下检索意图。确认后，系统才会基于该意图生成各检索来源的实际检索式和检索参数。",
    }


async def paper_search_confirm_node(
    state: Mapping[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    resume_value = _interrupt_with_config(
        config,
        _build_confirmation_payload(state),
    )

    if not isinstance(resume_value, dict):
        return {
            "stage": "blocked",
            "status": "blocked",
            "confirmation_required": False,
            "confirmation_decision": None,
            "error": "确认恢复参数必须是对象。",
        }

    decision = resume_value.get("decision")
    comment = resume_value.get("comment")

    if decision not in {"approved", "rejected"}:
        return {
            "stage": "blocked",
            "status": "blocked",
            "confirmation_required": False,
            "confirmation_decision": None,
            "error": "确认结果必须是 approved 或 rejected。",
        }

    if decision == "rejected":
        return {
            "stage": "blocked",
            "status": "blocked",
            "confirmation_required": False,
            "confirmation_decision": "rejected",
            "warnings": [
                *state.get("warnings", []),
                (
                    "用户拒绝执行论文检索任务。"
                    if comment is None
                    else f"用户拒绝执行论文检索任务：{comment}"
                ),
            ],
            "error": None,
        }

    submitted_understanding = resume_value.get(
        "query_understanding",
        state.get("query_understanding"),
    )

    submitted_search_tag = resume_value.get(
        "search_tag",
        state.get("search_tag"),
    )

    try:
        understanding = PromptUnderstandingArgs.model_validate(submitted_understanding)
        search_tag = SearchTagArgs.model_validate(submitted_search_tag)
        search_tag = with_derived_year_tag(search_tag, understanding)
    except Exception as exc:
        return {
            "stage": "blocked",
            "status": "blocked",
            "confirmation_required": False,
            "confirmation_decision": "approved",
            "error": f"确认后的检索条件无效：{exc}",
        }

    return {
        "stage": "creating_task",
        "status": "running",
        "confirmation_required": False,
        "confirmation_decision": "approved",
        "query_understanding": understanding.model_dump(mode="json"),
        "search_tag": search_tag.model_dump(mode="json"),
        "error": None,
    }