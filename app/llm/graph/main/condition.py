from app.llm.graph.main.state import (
    MainAgentState,
)


def after_solve(
    state: MainAgentState,
):
    if not state.get("pending_tool_calls"):
        return "final"
    return "dispatch"

def after_confirm(
    state: MainAgentState,
):
    return "dispatch" if state.get("active_tool_call") is None else _active_target(state)


def after_tool(state: MainAgentState) -> str:
    return "dispatch"


def after_dispatch(state: MainAgentState) -> str:
    if state.get("active_tool_call") is None:
        return "solve"
    return _active_target(state)


def _active_target(state: MainAgentState) -> str:
    call = state["active_tool_call"]
    if call["requires_confirmation"]:
        return "confirm"
    if call["kind"] == "subagent":
        return "subagent"
    return "tool"

