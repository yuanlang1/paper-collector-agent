from __future__ import annotations

import hashlib
import json
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.llm.graph.main.action_messages import build_action_result_message
from app.llm.graph.main.schemas import (
    ActionExecutionRecordState,
    ActionResultState,
)
from app.llm.graph.main.state import MainAgentState
from app.llm.tool_factory import ToolFactory
from app.llm.tools.registry import TOOL_BY_NAME


def arguments_digest(
    value: dict[str, Any],
) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def append_history(
    state: MainAgentState,
    record: ActionExecutionRecordState,
) -> list[ActionExecutionRecordState]:
    return [*state.get("action_history", []), record][-20:]


async def tool_node(
    state: MainAgentState,
    config: RunnableConfig,
) -> dict:
    pending = state.get("pending_action")

    if pending is None or pending["action"] != "tool":
        return {
            "error": "tool 节点缺少有效的工具动作",
            "run_status": "failed",
        }

    tool_name = pending["tool_name"]
    assert tool_name is not None

    request = {
        "name": tool_name,
        "arguments": pending["tool_arguments"],
    }

    spec = TOOL_BY_NAME.get(request["name"])

    if spec is None:
        result: ActionResultState = {
            "action_id": pending["action_id"],
            "action_type": "tool",
            "name": request["name"],
            "status": "error",
            "summary": "工具未注册。",
            "data": {},
            "artifact_refs": [],
            "retryable": False,
            "error_code": "UNKNOWN_TOOL",
            "error_message": request["name"],
        }
    else:
        db = config.get("configurable", {}).get("db")
        

        if db is None:
            result = {
                "action_id": pending["action_id"],
                "action_type": "tool",
                "name": request["name"],
                "status": "error",
                "summary": "工具执行缺少数据库会话。",
                "data": {},
                "artifact_refs": [],
                "retryable": False,
                "error_code": "DB_SESSION_MISSING",
                "error_message": None,
            }
        else:
            try:
                tools = ToolFactory().build(
                    db=db,
                    allowed_tools=[request["name"]],
                )
                tool = next(
                    item
                    for item in tools
                    if item.name == request["name"]
                )

                raw = await tool.ainvoke(request["arguments"])

                ok = bool(raw.get("ok")) if isinstance(raw, dict) else True
                
                artifact_refs = raw.get("artifact_refs", []) if isinstance(raw, dict) else []

                result = {
                    "action_id": pending["action_id"],
                    "action_type": "tool",
                    "name": request["name"],
                    "status": "success" if ok else "error",
                    "summary": (
                        "工具执行成功。"
                        if ok
                        else str(raw.get("error") or "工具执行失败")
                    ),
                    "data": (
                        raw
                        if isinstance(raw, dict)
                        else {"result": raw}
                    ),
                    "artifact_refs": [
                        str(item)
                        for item in artifact_refs
                    ],
                    "retryable": False,
                    "error_code": None,
                    "error_message": None if ok else str(raw.get("error"))
                }

            except Exception as exc:
                result = {
                    "action_id": pending["action_id"],
                    "action_type": "tool",
                    "name": request["name"],
                    "status": "error",
                    "summary": "工具执行异常。",
                    "data": {},
                    "artifact_refs": [],
                    "retryable": False,
                    "error_code": type(exc).__name__,
                    "error_message": str(exc),
                }

    record: ActionExecutionRecordState = {
        "action_id": pending["action_id"],
        "action_type": "tool",
        "name": request["name"],
        "arguments_digest": arguments_digest(request["arguments"]),
        "status": result["status"],
        "summary": result["summary"],
    }

    artifact_refs = list(
        dict.fromkeys(
            [
                *state.get("artifact_refs", []),
                *result["artifact_refs"],
            ]
        )
    )

    return {
        "last_action_result": result,
        "action_history": append_history(state, record),
        "artifact_refs": artifact_refs,
        "messages": [build_action_result_message(result)],
        "pending_action": None,
        "solve_decision": None,
        "confirmation": None,
    }