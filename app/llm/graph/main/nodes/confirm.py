from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import set_config_context
from langgraph.types import interrupt

from app.llm.graph.main.state import MainAgentState
from app.llm.graph.main.nodes.tool import build_action_result_update
from app.llm.subagents.registry import SubAgentRegistry


def _interrupt_with_config(
    config: RunnableConfig,
    payload: dict[str, Any],
) -> Any:
    with set_config_context(config) as context:
        return context.run(interrupt, payload)


async def confirm_node(
    state: MainAgentState,
    config: RunnableConfig,
    *,
    subagent_registry: SubAgentRegistry,
) -> dict:
    call = state["active_tool_call"]
    spec = (
        subagent_registry.get_spec(call["name"])
        if call.get("kind") == "subagent"
        else None
    )
    resume_value = _interrupt_with_config(
        config,
        {
            "action_id": call["id"],
            "action_type": call["kind"],
            "kind": call["kind"],
            "name": call["name"],
            "display_name": (
                spec.display_name
                if spec and spec.display_name
                else call["name"]
            ),
            "summary": (
                spec.confirmation_summary
                if spec and spec.confirmation_summary
                else "将执行此操作。"
            ),
            "requires_confirmation": True,
            "status": "pending",
        },
    )
    if resume_value.get("decision") == "approved":
        return {"active_tool_call": {**call, "requires_confirmation": False}}

    return build_action_result_update(
        call=call,
        status="rejected",
        summary="用户拒绝执行该操作。",
        error_code="USER_REJECTED",
    )
