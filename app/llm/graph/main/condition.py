from app.llm.graph.main.state import (
    MainAgentState,
)


def _pending_target(state: MainAgentState) -> str:
    pending = state.get("pending_action")
    if pending is None:
        return "solve"
    if pending["action"] == "subagent":
        if pending["subagent_name"] != "paper_search_agent":
            raise ValueError(
                f"Unsupported native subagent: "
                f"{pending['subagent_name']}"
            )
        return "paper_search"
    return pending["action"]


def after_solve(
    state: MainAgentState,
):
    decision = state.get("solve_decision")
    pending = state.get("pending_action")

    if (
        decision is not None
        and decision["action"] == "direct"
    ):
        return "final"

    if pending is None:
        return "solve"

    if pending["requires_confirmation"]:
        return "confirm"

    return _pending_target(state)

def after_confirm(
    state: MainAgentState,
):
    if state.get("run_status") == "failed":
        return "final"

    confirmation = state.get("confirmation")
    pending = state.get("pending_action")

    if (
        confirmation is None
        or confirmation["decision"] == "rejected"
        or pending is None
    ):
        return "solve"

    return _pending_target(state)


def after_tool(state: MainAgentState) -> str:
    if state.get("run_status") == "failed":
        return "final"
    return "solve"


def after_prepare_paper_search(
    state: MainAgentState,
) -> str:
    if state.get("paper_search_handoff") is not None:
        return "complete"
    return "run"

