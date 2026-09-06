import json

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from app.llm.graph.main.state import MainAgentState
from app.llm.tools.registry import ToolRegistry


async def tool_node(
    state: MainAgentState,
    config: RunnableConfig,
    *,
    tool_registry: ToolRegistry,
) -> dict:
    del config
    call = state["active_tool_call"]
    result = await tool_registry.execute(call["name"], call["args"])

    return {
        "messages": [
            ToolMessage(
                tool_call_id=call["id"],
                name=call["name"],
                content=json.dumps(result, ensure_ascii=False, default=str),
            )
        ],
        "active_tool_call": None,
        "last_action_result": {
            "action_id": call["id"],
            "action_type": "tool",
            "name": call["name"],
            "status": "success" if result.get("ok", True) else "error",
            "summary": "tool completed",
            "data": result,
            "artifact_refs": result.get("artifact_refs", []),
            "retryable": False,
            "error_code": result.get("error"),
            "error_message": result.get("message"),
        },
        "artifact_refs": result.get("artifact_refs", []),
    }
