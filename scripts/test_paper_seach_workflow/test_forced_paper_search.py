import unittest

from app.api.schemas.agent import ChatRequest
from app.llm.graph.main.nodes.solve import SolveNode
from app.llm.graph.main.schemas import SolveDecision
from app.llm.graph.workflows.paper_search.nodes.build_search_tag import (
    BuildSearchTagNode,
)
from app.llm.graph.workflows.paper_search.nodes.intent import (
    IntentUnderstandingNode,
)
from app.llm.subagents.paper_search.contracts import PaperSearchConstraints
from app.llm.subagents.paper_search.agent import PaperSearchDelegation
from app.llm.subagents.contracts import SubAgentSpec
from app.llm.subagents.registry import SubAgentRegistry
from app.llm.tools.task_tools.search_task.args import (
    PromptUnderstandingArgs,
    SearchTagArgs,
)


class _FailingSolver:
    async def ainvoke(self, _messages):
        raise AssertionError("forced subagent must skip the solver")


class _DirectSolver:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, _messages):
        self.calls += 1
        return type(
            "Decision",
            (),
            {
                "to_state": lambda _self: {
                    "action": "direct",
                    "answer": "done",
                    "tool_name": None,
                    "tool_arguments": {},
                    "subagent_name": None,
                    "subagent_input": {},
                    "confirmation_hint": None,
                    "decision_reason": "result available",
                }
            },
        )()


class _BlankDirectSolver:
    async def ainvoke(self, _messages):
        return SolveDecision.model_validate(
            {
                "action": "direct",
                "decision_reason": "return the result to the user",
            }
        )


class _IntentModel:
    async def ainvoke(self, _messages):
        return PromptUnderstandingArgs(
            topic="RAG",
            intent="benchmark",
            reasoning="test",
        )


class _TagModel:
    async def ainvoke(self, _messages):
        return SearchTagArgs()


class ForcedPaperSearchTests(unittest.IsolatedAsyncioTestCase):
    def test_chat_request_accepts_forced_paper_search(self) -> None:
        request = ChatRequest.model_validate(
            {
                "message": "Search RAG papers",
                "subagent": {
                    "name": "paper_search_agent",
                    "constraints": {
                        "year_from": 2023,
                        "year_to": 2026,
                        "sources": ["arXiv", "DBLP"],
                    },
                },
            }
        )

        self.assertEqual(request.subagent.constraints.year_from, 2023)

    async def test_forced_subagent_is_consumed_before_solver_runs(self) -> None:
        node = SolveNode(
            solver=_FailingSolver(),
            subagent_registry=SubAgentRegistry(
                [
                    SubAgentSpec(
                        name="paper_search_agent",
                        description="test",
                        input_model=PaperSearchDelegation,
                    )
                ]
            ),
        )
        result = await node(
            {
                "forced_subagent": {
                    "name": "paper_search_agent",
                    "input": {"prompt": "RAG", "constraints": {}},
                },
                "action_rounds": 0,
                "max_action_rounds": 8,
            }
        )

        self.assertIsNone(result["forced_subagent"])
        self.assertEqual(
            result["pending_action"]["subagent_name"],
            "paper_search_agent",
        )

    async def test_consumed_force_returns_to_normal_solver_behavior(self) -> None:
        solver = _DirectSolver()
        node = SolveNode(
            solver=solver,
            subagent_registry=SubAgentRegistry([]),
        )

        result = await node(
            {
                "forced_subagent": None,
                "action_rounds": 1,
                "max_action_rounds": 8,
                "messages": [],
                "last_action_result": {"status": "success"},
                "action_history": [],
            }
        )

        self.assertEqual(solver.calls, 1)
        self.assertEqual(result["solve_decision"]["action"], "direct")

    async def test_constraints_override_years_and_sources(self) -> None:
        constraints = PaperSearchConstraints(
            year_from=2020,
            year_to=2023,
            sources=["DBLP"],
        ).model_dump(mode="json")
        intent = await IntentUnderstandingNode(model=_IntentModel())(
            {
                "original_prompt": "RAG papers",
                "paper_search_constraints": constraints,
            }
        )
        tag = await BuildSearchTagNode(model=_TagModel())(
            {
                "original_prompt": "RAG papers",
                "query_understanding": intent["query_understanding"],
                "paper_search_constraints": constraints,
                "warnings": [],
            }
        )

        self.assertEqual(intent["query_understanding"]["yearFrom"], 2020)
        self.assertEqual(intent["query_understanding"]["yearTo"], 2023)
        self.assertEqual(tag["search_tag"]["sourceTag"], ["DBLP"])
        self.assertEqual(tag["search_tag"]["yearTag"], 0)

    async def test_blank_direct_answer_uses_action_result_summary(self) -> None:
        node = SolveNode(
            solver=_BlankDirectSolver(),
            subagent_registry=SubAgentRegistry([]),
        )

        result = await node(
            {
                "action_rounds": 0,
                "max_action_rounds": 8,
                "messages": [],
                "last_action_result": {"summary": "论文检索完成，推荐 8 篇。"},
                "action_history": [],
            }
        )

        self.assertEqual(
            result["solve_decision"]["answer"],
            "论文检索完成，推荐 8 篇。",
        )

    async def test_blank_direct_answer_uses_default_without_action_result(self) -> None:
        node = SolveNode(
            solver=_BlankDirectSolver(),
            subagent_registry=SubAgentRegistry([]),
        )

        result = await node(
            {
                "action_rounds": 0,
                "max_action_rounds": 8,
                "messages": [],
                "last_action_result": None,
                "action_history": [],
            }
        )

        self.assertEqual(
            result["solve_decision"]["answer"],
            "任务已完成，请查看本次执行结果。",
        )
