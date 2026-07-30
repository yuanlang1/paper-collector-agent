from __future__ import annotations

from pydantic import ValidationError

from app.llm.graph.main.action_messages import build_action_result_message
from app.llm.graph.main.nodes.tool import append_history, arguments_digest
from app.llm.graph.main.schemas import (
    ActionExecutionRecordState,
    ActionResultState,
)
from app.llm.graph.main.state import MainAgentState
from app.llm.subagents.paper_search.contracts import (
    PaperSearchDelegation,
    PaperSearchHandoff,
)


async def prepare_paper_search_node(
    state: MainAgentState,
) -> dict:
    pending = state.get("pending_action")
    if (
        pending is None
        or pending["action"] != "subagent"
        or pending["subagent_name"] != "paper_search_agent"
    ):
        return {
            "paper_search_request": None,
            "paper_search_handoff": {
                "status": "error",
                "summary": "缺少有效的论文检索动作。",
                "data": {},
                "artifact_refs": [],
                "retryable": False,
                "error_code": "INVALID_PAPER_SEARCH_ACTION",
                "error_message": None,
            },
        }

    try:
        request = PaperSearchDelegation.model_validate(
            pending["subagent_input"]
        )
    except ValidationError as exc:
        return {
            "paper_search_request": None,
            "paper_search_handoff": {
                "status": "error",
                "summary": "论文检索请求无效。",
                "data": {},
                "artifact_refs": [],
                "retryable": False,
                "error_code": "INVALID_PAPER_SEARCH_REQUEST",
                "error_message": str(exc),
            },
            "paper_search_action_id": pending["action_id"],
        }

    return {
        "paper_search_request": request.model_dump(mode="json"),
        "paper_search_handoff": None,
        "paper_search_action_id": pending["action_id"],
    }


async def complete_paper_search_node(
    state: MainAgentState,
) -> dict:
    pending = state.get("pending_action")
    action_id = (
        pending["action_id"]
        if pending is not None
        else state.get("paper_search_action_id") or "invalid"
    )
    name = (
        pending["subagent_name"]
        if pending is not None
        else "paper_search_agent"
    )
    arguments = (
        pending["subagent_input"]
        if pending is not None
        else {}
    )

    try:
        handoff = PaperSearchHandoff.model_validate(
            state.get("paper_search_handoff")
        )
        handoff_state = handoff.model_dump(mode="json")
        result: ActionResultState = {
            "action_id": action_id,
            "action_type": "subagent",
            "name": str(name),
            **handoff_state,
        }
    except ValidationError as exc:
        result = {
            "action_id": action_id,
            "action_type": "subagent",
            "name": str(name),
            "status": "error",
            "summary": "论文检索子图未返回有效结果。",
            "data": {},
            "artifact_refs": [],
            "retryable": False,
            "error_code": "INVALID_PAPER_SEARCH_HANDOFF",
            "error_message": str(exc),
        }

    record: ActionExecutionRecordState = {
        "action_id": action_id,
        "action_type": "subagent",
        "name": result["name"],
        "arguments_digest": arguments_digest(arguments),
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
        "paper_search_request": None,
        "paper_search_handoff": None,
        "paper_search_action_id": None,
    }
