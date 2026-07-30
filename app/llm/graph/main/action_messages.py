import json

from langchain_core.messages import (
    AIMessage,
    ToolMessage,
)

from app.llm.graph.main.schemas import (
    ActionResultState,
    PendingActionState,
)


def build_decision_call_message(
    pending: PendingActionState,
) -> AIMessage:
    return AIMessage(
        content = "",
        tool_calls = [
            {
                "name": "SolveDecision",
                "args": {
                    "action": pending["action"],
                    "tool_name": pending["tool_name"],
                    "tool_arguments": pending["tool_arguments"],
                    "subagent_name": pending["subagent_name"],
                    "subagent_input": pending["subagent_input"],
                    "decision_reason": pending["decision_reason"],
                },
                "id": pending["action_id"],
                "type": "tool_call",
            }
        ],
    )


def build_action_result_message(
    result: ActionResultState,
) -> ToolMessage:
    return ToolMessage(
        tool_call_id = result["action_id"],
        name = "SolveDecision",
        content = json.dumps(
            {
                "executed_action": result["action_type"],
                "executed_name": result["name"],
                "status": result["status"],
                "summary": result["summary"],
                "data": result["data"],
                "artifact_refs": result["artifact_refs"],
                "retryable": result["retryable"],
                "error_code": result["error_code"],
                "error_message": result["error_message"],
            },
            ensure_ascii = False,
            default = str,
        ),
    )
