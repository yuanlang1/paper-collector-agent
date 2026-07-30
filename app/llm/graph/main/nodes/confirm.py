from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import set_config_context
from langgraph.types import interrupt

from app.llm.graph.main.action_messages import build_action_result_message
from app.llm.graph.main.state import MainAgentState


def _interrupt_with_config(
    config: RunnableConfig,
    payload: dict[str, Any],
) -> Any:
    with set_config_context(config) as context:
        return context.run(
            interrupt,
            payload,
        )


async def confirm_node(
    state: MainAgentState,
    config: RunnableConfig,
) -> dict:
    pending = state.get("pending_action")

    if pending is None:
        return {
            "error": "confirm 节点缺少 pending_action",
            "run_status": "failed",
        }

    resume_value = _interrupt_with_config(
        config,
        {
            "type": "action_confirmation",
            "action_id": pending["action_id"],
            "action": pending["action"],
            "tool_name": pending["tool_name"],
            "tool_arguments": pending["tool_arguments"],
            "subagent_name": pending["subagent_name"],
            "subagent_input": pending["subagent_input"],
            "message": pending["confirmation_message"] or "是否允许继续执行？"
        },
    )

    if not isinstance(resume_value, dict):
        return {
            "error": "确认恢复参数必须是对象。",
            "run_status": "failed",
        }

    decision = resume_value.get("decision")
    comment = resume_value.get("comment")

    if decision not in {"approved", "rejected"}:
        return {
            "error": "确认结果必须是 approved 或 rejected。",
            "run_status": "failed",
        }

    confirmation = {
        "action_id": pending["action_id"],
        "decision": decision,
        "comment": str(comment) if comment is not None else None
    }

    if decision == "approved":
        return {
            "confirmation": confirmation,
            "run_status": "running",
        }

    name = (
        pending["tool_name"]
        if pending["action"] == "tool" else pending["subagent_name"]
    )
    assert name is not None

    rejected_result = {
        "action_id": pending["action_id"],
        "action_type": pending["action"],
        "name": name,
        "status": "rejected",
        "summary": "用户拒绝执行该动作。",
        "data": {},
        "artifact_refs": [],
        "retryable": False,
        "error_code": "USER_REJECTED",
        "error_message":  str(comment) if comment is not None else None
    }

    return {
        "confirmation": confirmation,
        "pending_action": None,
        "solve_decision": None,
        "last_action_result": rejected_result,
        "messages": [
            build_action_result_message(rejected_result)
        ],
        "run_status": "running",
    }
