from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from langchain_core.messages import ToolMessage


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
