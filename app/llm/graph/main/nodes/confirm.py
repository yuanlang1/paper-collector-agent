import json
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import set_config_context
from langgraph.types import interrupt

from app.llm.graph.main.state import MainAgentState


def _interrupt_with_config(
    config: RunnableConfig,
    payload: dict[str, Any],
) -> Any:
    with set_config_context(config) as context:
        return context.run(interrupt, payload)


async def confirm_node(
    state: MainAgentState,
    config: RunnableConfig,
) -> dict:
    call = state["active_tool_call"]
    resume_value = _interrupt_with_config(
        config,
        {
            "action_id": call["id"],
            "action_type": call["kind"],
            "kind": call["kind"],
            "name": call["name"],
            "display_name": {
                "paper_search_agent": "论文检索子代理",
                "task_review_agent": "文献综述子代理",
            }.get(call["name"], call["name"]),
            "summary": {
                "paper_search_agent": "将从多个来源检索、推荐并保存论文。",
                "task_review_agent": "将分析已有文献并生成综述建议。",
            }.get(call["name"], "将执行此操作。"),
            "requires_confirmation": True,
            "status": "pending",
        },
    )
    if resume_value.get("decision") == "approved":
        return {"active_tool_call": {**call, "requires_confirmation": False}}

    return {
        "messages": [
            ToolMessage(
                tool_call_id=call["id"],
                name=call["name"],
                content=json.dumps({"ok": False, "error": "USER_REJECTED"}),
            )
        ],
        "active_tool_call": None,
    }
