import json
from collections.abc import Mapping
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from app.llm.graph.main.state import MainAgentState
from app.llm.artifacts.access import ArtifactAccessService
from app.database import SessionLocal
from app.llm.streaming.notify import langgraph_notifier
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
    artifact_access_service: ArtifactAccessService | None = None,
) -> dict:
    call = state["active_tool_call"]
    notify = langgraph_notifier(config).scoped(
        source="tool",
        action_id=str(call["id"]),
        tool_name=str(call["name"]),
    )
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
                artifact_access=artifact_access_service,
                _notify=notify,
            ),
        )
    finally:
        db.close()

    return build_action_result_update(
        call=call,
        status="error" if result.is_error else "success",
        summary="tool failed" if result.is_error else "tool completed",
        data={"content": result.content},
        error_code=result.error_type,
        error_message=result.content if result.is_error else None,
    )
