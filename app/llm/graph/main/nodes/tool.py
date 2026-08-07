import json

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from app.llm.graph.main.state import MainAgentState
from app.llm.tool_factory import ToolFactory
from app.llm.tools.registry import TOOL_BY_NAME


async def tool_node(
    state: MainAgentState,
    config: RunnableConfig,
) -> dict:
    call = state["active_tool_call"]

    if call["name"] not in TOOL_BY_NAME:
        result = {"ok": False, "error": "UNKNOWN_TOOL"}
    else:
        tools = ToolFactory().build(
            db=config["configurable"]["db"],
            allowed_tools=[call["name"]],
        )
        tool = next(
            item
            for item in tools
            if item.name == call["name"]
        )

        try:
            raw = await tool.ainvoke(call["args"])
            result = (
                raw
                if isinstance(raw, dict)
                else {"result": raw}
            )
        except Exception as exc:
            result = {
                "ok": False,
                "error": type(exc).__name__,
                "message": str(exc),
            }

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
