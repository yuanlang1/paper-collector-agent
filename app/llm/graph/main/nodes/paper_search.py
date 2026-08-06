from __future__ import annotations

import json

from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from app.llm.graph.main.state import MainAgentState
from app.llm.subagents.paper_search.contracts import (
    PaperSearchDelegation,
    PaperSearchHandoff,
)


async def prepare_paper_search_node(state: MainAgentState) -> dict:
    call = state["active_tool_call"]
    try:
        request = PaperSearchDelegation.model_validate(call["args"])
    except ValidationError as exc:
        return {
            "paper_search_handoff": {
                "status": "error",
                "summary": "invalid paper search request",
                "data": {},
                "artifact_refs": [],
                "retryable": False,
                "error_code": "INVALID_PAPER_SEARCH_REQUEST",
                "error_message": str(exc),
            },
            "paper_search_tool_call_id": call["id"],
        }

    return {
        "paper_search_request": request.model_dump(mode="json"),
        "paper_search_handoff": None,
        "paper_search_tool_call_id": call["id"],
    }


async def complete_paper_search_node(state: MainAgentState) -> dict:
    handoff = PaperSearchHandoff.model_validate(state["paper_search_handoff"])
    result = handoff.model_dump(mode="json")
    tool_call_id = state["paper_search_tool_call_id"]

    return {
        "messages": [
            ToolMessage(
                tool_call_id=tool_call_id,
                name="paper_search_agent",
                content=json.dumps(result, ensure_ascii=False),
            )
        ],
        "active_tool_call": None,
        "last_action_result": {
            "action_id": tool_call_id,
            "action_type": "subagent",
            "name": "paper_search_agent",
            **result,
        },
        "paper_search_request": None,
        "paper_search_handoff": None,
        "paper_search_tool_call_id": None,
    }
