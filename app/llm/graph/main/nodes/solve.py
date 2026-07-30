import json
from uuid import uuid4

from langchain_core.messages import SystemMessage

from app.llm.graph.main.action_messages import build_decision_call_message
from app.llm.graph.main.schemas import (
    ActionResultState,
    PendingActionState,
    SolveDecision,
    SolveDecisionState,
)
from app.llm.graph.main.state import MainAgentState
from app.llm.tools.registry import TOOL_BY_NAME, tool_descriptions


SOLVE_SYSTEM_PROMPT = """
你是 Agent 的下一步动作决策器。每次只能选择一个动作：

1. direct：现有信息已足够回答用户。
2. tool：只需要一个原子工具。
3. subagent：需要多阶段、领域型任务时委派专业 Agent。

当用户要求跨数据源检索、筛选、富化、推荐或保存论文时，使用
paper_search_agent。不要把它当成工具或 workflow。

规则：
- action=direct 时填写 answer。
- action=tool 时填写 tool_name 和 tool_arguments。
- action=subagent 时填写 subagent_name 和 subagent_input。
- 最近 ToolMessage 的 data 已足够回答时，必须选择 direct。
- 仅当结果明确缺少必要信息时，才能继续选择 tool 或 subagent。
- 不得对 retryable=false 的成功动作以相同参数重复执行。
"""


DIRECT_ANSWER_REQUIREMENT = """
When action is direct, answer is required. Populate answer with a non-empty,
user-facing final response. Never put the final response only in
decision_reason.
""".strip()


class SolveNode:
    def __init__(self, *, solver, subagent_registry):
        self.solver = solver
        self.subagent_registry = subagent_registry

    async def __call__(self, state: MainAgentState) -> dict:
        rounds = state.get("action_rounds", 0)
        max_rounds = state.get("max_action_rounds", 8)
        if rounds >= max_rounds:
            decision: SolveDecisionState = {
                "action": "direct",
                "answer": "本次执行已达到最大动作轮数，已停止继续调用。",
                "tool_name": None,
                "tool_arguments": {},
                "subagent_name": None,
                "subagent_input": {},
                "confirmation_hint": None,
                "decision_reason": "action round limit reached",
            }
            return {"solve_decision": decision, "pending_action": None}

        forced_subagent = state.get("forced_subagent")
        if forced_subagent is not None:
            decision: SolveDecisionState = {
                "action": "subagent",
                "answer": None,
                "tool_name": None,
                "tool_arguments": {},
                "subagent_name": forced_subagent["name"],
                "subagent_input": forced_subagent["input"],
                "confirmation_hint": "是否开始论文检索？",
                "decision_reason": "frontend forced subagent",
            }
            pending_or_error = self._prepare_action(decision)
            if "status" in pending_or_error:
                return {
                    "forced_subagent": None,
                    "solve_decision": None,
                    "pending_action": None,
                    "last_action_result": pending_or_error,
                    "action_rounds": rounds + 1,
                }
            return {
                "forced_subagent": None,
                "solve_decision": decision,
                "pending_action": pending_or_error,
                "confirmation": None,
                "messages": [build_decision_call_message(pending_or_error)],
                "action_rounds": rounds + 1,
            }

        output: SolveDecision = await self.solver.ainvoke([
            SystemMessage(
                content=(
                    f"{SOLVE_SYSTEM_PROMPT}\n\n"
                    f"{DIRECT_ANSWER_REQUIREMENT}\n\n"
                    f"{self._build_context(state)}"
                )
            ),
            *state.get("messages", []),
        ])
        decision = output.to_state()
        if decision["action"] == "direct" and not decision["answer"]:
            decision["answer"] = self._direct_answer_fallback(state)
            decision["decision_reason"] = (
                "model omitted direct answer; used action-result fallback"
            )
        if decision["action"] == "direct":
            return {
                "solve_decision": decision,
                "pending_action": None,
                "action_rounds": rounds + 1,
            }

        pending_or_error = self._prepare_action(decision)
        if "status" in pending_or_error:
            return {
                "solve_decision": None,
                "pending_action": None,
                "last_action_result": pending_or_error,
                "action_rounds": rounds + 1,
            }
        return {
            "solve_decision": decision,
            "pending_action": pending_or_error,
            "confirmation": None,
            "messages": [build_decision_call_message(pending_or_error)],
            "action_rounds": rounds + 1,
        }

    def _prepare_action(
        self, decision: SolveDecisionState
    ) -> PendingActionState | ActionResultState:
        if decision["action"] == "tool":
            name = decision["tool_name"]
            spec = TOOL_BY_NAME.get(name)
            if spec is None:
                return self._unknown_result("tool", name, "UNKNOWN_TOOL")
            return {
                "action_id": f"action_{uuid4().hex}", "action": "tool",
                "tool_name": name, "tool_arguments": decision["tool_arguments"],
                "subagent_name": None, "subagent_input": {},
                "confirmation_hint": decision["confirmation_hint"],
                "decision_reason": decision["decision_reason"],
                "requires_confirmation": spec.requires_confirmation,
                "confirmation_message": decision["confirmation_hint"] or f"是否允许执行工具 {name}？",
            }

        if decision["action"] != "subagent":
            raise ValueError(f"Unsupported action: {decision['action']}")
        name = decision["subagent_name"]
        spec = self.subagent_registry.get(name)
        if spec is None:
            return self._unknown_result("subagent", name, "UNKNOWN_SUBAGENT")
        return {
            "action_id": f"action_{uuid4().hex}", "action": "subagent",
            "tool_name": None, "tool_arguments": {},
            "subagent_name": name, "subagent_input": decision["subagent_input"],
            "confirmation_hint": decision["confirmation_hint"],
            "decision_reason": decision["decision_reason"],
            "requires_confirmation": spec.requires_confirmation,
            "confirmation_message": decision["confirmation_hint"] or f"是否允许委派 {name}？",
        }

    @staticmethod
    def _unknown_result(action_type: str, name: str | None, error_code: str) -> ActionResultState:
        return {
            "action_id": "invalid", "action_type": action_type, "name": str(name),
            "status": "error", "summary": f"{action_type} is not registered", "data": {},
            "artifact_refs": [], "retryable": False, "error_code": error_code,
            "error_message": str(name),
        }

    @staticmethod
    def _direct_answer_fallback(state: MainAgentState) -> str:
        result = state.get("last_action_result")
        if isinstance(result, dict):
            summary = result.get("summary")
            if isinstance(summary, str) and summary.strip():
                return summary.strip()
        return "任务已完成，请查看本次执行结果。"

    def _build_context(self, state: MainAgentState) -> str:
        last_result = json.dumps(state.get("last_action_result"), ensure_ascii=False)
        history = json.dumps(state.get("action_history", [])[-6:], ensure_ascii=False)
        subagents = self.subagent_registry.descriptions() or "暂无已注册 subagent"
        return (
            f"可用工具：\n{tool_descriptions()}\n\n"
            f"可用 subagent：\n{subagents}\n\n"
            f"最近一次执行结果：\n{last_result}\n\n"
            f"最近动作历史：\n{history}"
        )
