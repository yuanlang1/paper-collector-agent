from app.llm.graph.main.state import MainAgentState


async def dispatch_tool_call_node(state: MainAgentState) -> dict:
    if state.get("active_tool_call") is not None:
        return {}

    calls = state.get("pending_tool_calls", [])
    if not calls:
        return {}

    return {
        "active_tool_call": calls[0],
        "pending_tool_calls": calls[1:],
    }
