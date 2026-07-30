from langchain_core.messages import (
    AIMessage,
)

from app.llm.graph.main.schemas import (
    RunStatus,
)
from app.llm.graph.main.state import (
    MainAgentState,
)


async def final_node(
    state: MainAgentState,
) -> dict:
    decision = state.get("solve_decision")

    if (
        state.get("run_status") == "failed"
        or decision is None
        or decision["action"] != "direct"
    ):
        error = state.get("error") or "final 节点缺少 direct 结果"

        return {
            "reply": f"任务执行失败：{error}",
            "messages": [
                AIMessage(content = (f"任务执行失败：{error}"))
            ],
            "run_status": "failed",
            "error": error,
            "pending_action": None,
            "solve_decision": None,
            "confirmation": None,
        }

    reply = decision["answer"].strip()

    return {
        "reply": reply,
        "messages": [AIMessage(content = reply)],
        "run_status": "completed",
        "pending_action": None,
        "confirmation": None,
        "error": None,
    }
