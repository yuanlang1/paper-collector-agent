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
            "type": "tool_confirmation",
            "tool_call_id": call["id"],
            "tool_name": call["name"],
            "tool_arguments": call["args"],
            "message": f"Allow {call['name']}?",
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
