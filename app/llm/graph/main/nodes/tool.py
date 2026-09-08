import json
from collections.abc import Mapping
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from app.llm.graph.main.state import MainAgentState
from app.database import SessionLocal
from app.llm.tools.registry import ToolExecutionContext, ToolRegistry


ActionStatus = Literal["success", "partial", "error", "rejected"]


def build_action_result_update(
    *,
    call: Mapping[str, Any],
    status: ActionStatus,
    summary: str,
    data: Mapping[str, Any] | None = None,
    artifact_refs: list[str] | None = None,
    retryable: bool = False,
    error_code: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Build the shared state update for a completed tool or subagent call."""
    action_id = str(call["id"])
    action_name = str(call["name"])
    result = {
        "status": status,
        "summary": summary,
        "data": dict(data or {}),
        "artifact_refs": list(artifact_refs or []),
        "retryable": retryable,
        "error_code": error_code,
        "error_message": error_message,
    }
    action_result = {
        "action_id": action_id,
        "action_type": str(call.get("kind") or "tool"),
        "name": action_name,
        **result,
    }
    return {
        "messages": [
            ToolMessage(
                tool_call_id=action_id,
                name=action_name,
                content=json.dumps(result, ensure_ascii=False, default=str),
            )
        ],
        "active_tool_call": None,
        "last_action_result": action_result,
        "artifact_refs": result["artifact_refs"],
    }


async def tool_node(
    state: MainAgentState,
    config: RunnableConfig,
    *,
    tool_registry: ToolRegistry,
) -> dict:
    del config
    call = state["active_tool_call"]
    db = SessionLocal()
    try:
        result = await tool_registry.execute(
            call["name"],
            call["args"],
            context=ToolExecutionContext(
                db=db,
                user_id=str(state.get("user_id") or "0"),
                conversation_id=str(state.get("conversation_id") or ""),
                run_id=str(state.get("run_id") or ""),
            ),
        )
    finally:
        db.close()

    artifact_refs = result.get("artifact_refs")
    return build_action_result_update(
        call=call,
        status="success" if result.get("ok", True) else "error",
        summary="tool completed",
        data=result,
        artifact_refs=artifact_refs if isinstance(artifact_refs, list) else [],
        error_code=result.get("error"),
        error_message=result.get("message"),
    )
