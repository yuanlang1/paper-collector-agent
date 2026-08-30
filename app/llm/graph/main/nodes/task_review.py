from __future__ import annotations

import json

from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from app.llm.graph.main.state import MainAgentState
from app.llm.graph.main.nodes.safe_subgraph import build_subagent_error_handoff
from app.llm.subagents.task_review.contracts import (
    TaskReviewDelegation,
    TaskReviewHandoff,
)


async def prepare_task_review_node(state: MainAgentState) -> dict:
    call = state["active_tool_call"]
    try:
        request = TaskReviewDelegation.model_validate(call["args"])
    except ValidationError as exc:
        return {
            "task_review_handoff": {
                "status": "error",
                "summary": "invalid task review request",
                "data": {},
                "artifact_refs": [],
                "retryable": False,
                "error_code": "INVALID_TASK_REVIEW_REQUEST",
                "error_message": str(exc),
            },
            "task_review_tool_call_id": call["id"],
        }

    return {
        "task_review_request": request.model_dump(mode="json"),
        "task_review_handoff": None,
        "task_review_tool_call_id": call["id"],
    }


async def complete_task_review_node(state: MainAgentState) -> dict:
    tool_call_id = state["task_review_tool_call_id"]
    try:
        handoff = TaskReviewHandoff.model_validate(state["task_review_handoff"])
        result = handoff.model_dump(mode="json")
    except Exception as exc:
        result = build_subagent_error_handoff(
            subagent="task_review_agent",
            error_code="TASK_REVIEW_HANDOFF_INVALID",
            summary="文献综述工作流返回了无法处理的结果。",
            exception=exc,
        )

    return {
        "messages": [
            ToolMessage(
                tool_call_id=tool_call_id,
                name="task_review_agent",
                content=json.dumps(result, ensure_ascii=False),
            )
        ],
        "active_tool_call": None,
        "last_action_result": {
            "action_id": tool_call_id,
            "action_type": "subagent",
            "name": "task_review_agent",
            **result,
        },
        "task_review_request": None,
        "task_review_handoff": None,
        "task_review_tool_call_id": None,
    }
