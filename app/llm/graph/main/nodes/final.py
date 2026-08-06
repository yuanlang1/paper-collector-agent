from app.llm.graph.main.state import MainAgentState


async def final_node(state: MainAgentState) -> dict:
    if state.get("run_status") == "completed":
        return {
            "reply": state.get("reply", ""),
            "run_status": "completed",
            "error": None,
        }

    error = state.get("error") or "agent failed"
    return {
        "reply": f"任务执行失败：{error}",
        "run_status": "failed",
        "error": error,
    }
