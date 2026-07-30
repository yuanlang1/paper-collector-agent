import unittest
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.llm.graph.main.condition import (
    after_confirm,
    after_solve,
    after_tool,
)
from app.llm.graph.main.nodes.final import final_node
from app.llm.graph.main.workflow import build_main_agent_workflow
from app.llm.graph.workflows.paper_search.workflow import (
    build_paper_search_workflow,
)
from app.llm.subagents.paper_search.agent import (
    build_paper_search_agent_spec,
)
from app.llm.subagents.paper_search.contracts import (
    PaperSearchDelegation,
    build_paper_search_handoff,
)
from app.llm.subagents.registry import SubAgentRegistry


class MainGraphRegistrationTests(unittest.TestCase):
    def test_registers_paper_search_subagent(self) -> None:
        spec = build_paper_search_agent_spec()
        registry = SubAgentRegistry([spec])

        self.assertIs(registry.get("paper_search_agent"), spec)
        self.assertTrue(spec.requires_confirmation)
        request = spec.input_model.model_validate({"prompt": "RAG"})
        self.assertIsInstance(request, PaperSearchDelegation)

    def test_maps_terminal_workflow_state_to_subagent_handoff(self) -> None:
        handoff = build_paper_search_handoff({
            "stage": "completed",
            "progress": {
                "recommendation_kept": 3,
                "persistence_saved": 3,
            },
            "recommendation_manifest_artifact_ref": "artifact://run/recommend.json",
            "persisted_papers_manifest_artifact_ref": "artifact://run/persist.json",
        })

        self.assertEqual(handoff.status, "success")
        self.assertEqual(handoff.data["counts"]["recommended_papers"], 3)
        self.assertEqual(
            handoff.artifact_refs,
            [
                "artifact://run/recommend.json",
                "artifact://run/persist.json",
            ],
        )

    def test_maps_only_explicit_user_rejection_to_rejected(self) -> None:
        invalid_request = build_paper_search_handoff({
            "stage": "blocked",
            "status": "blocked",
            "confirmation_decision": None,
            "error": "invalid request",
        })
        user_rejection = build_paper_search_handoff({
            "stage": "blocked",
            "status": "blocked",
            "confirmation_decision": "rejected",
            "error": None,
        })

        self.assertEqual(invalid_request.status, "error")
        self.assertEqual(user_rejection.status, "rejected")

    def test_registers_compiled_paper_search_graph_as_a_node(self) -> None:
        async def node(_state, *_args):
            return {}

        paper_search_graph = build_paper_search_workflow(
            skip_confirmation=True,
        )
        graph = build_main_agent_workflow(
            solve_node=node,
            paper_search_graph=paper_search_graph,
        )

        registered = graph.get_graph().nodes["paper_search"]
        self.assertIs(registered.data, paper_search_graph)
        self.assertNotIn("subagent", graph.get_graph().nodes)

    def test_routes_paper_search_subagent_to_native_graph(self) -> None:
        route = after_solve({
            "solve_decision": {
                "action": "subagent",
            },
            "pending_action": {
                "action_id": "action_1",
                "action": "subagent",
                "tool_name": None,
                "tool_arguments": {},
                "subagent_name": "paper_search_agent",
                "subagent_input": {"prompt": "RAG"},
                "confirmation_hint": None,
                "decision_reason": "test",
                "requires_confirmation": False,
                "confirmation_message": None,
            },
        })

        self.assertEqual(route, "paper_search")

    def test_routes_failed_confirm_to_final(self) -> None:
        self.assertEqual(
            after_confirm({
                "run_status": "failed",
                "confirmation": None,
                "pending_action": None,
            }),
            "final",
        )

    def test_routes_failed_tool_to_final(self) -> None:
        self.assertEqual(
            after_tool({"run_status": "failed"}),
            "final",
        )

    def test_routes_rejected_confirmation_back_to_solve(self) -> None:
        self.assertEqual(
            after_confirm({
                "run_status": "running",
                "confirmation": {
                    "action_id": "action_1",
                    "decision": "rejected",
                    "comment": None,
                },
                "pending_action": None,
            }),
            "solve",
        )


class _NativePaperSearchState(TypedDict, total=False):
    paper_search_request: dict[str, Any] | None
    paper_search_handoff: dict[str, Any] | None


class MainGraphNativeSubgraphExecutionTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_invalid_confirmation_stays_failed(self) -> None:
        class Solve:
            calls = 0

            async def __call__(self, _state):
                self.calls += 1
                if self.calls > 1:
                    raise AssertionError(
                        "failed confirmation must not return to solve"
                    )
                return {
                    "solve_decision": {
                        "action": "tool",
                        "answer": None,
                        "tool_name": "get_current_time",
                        "tool_arguments": {},
                        "subagent_name": None,
                        "subagent_input": {},
                        "confirmation_hint": "confirm?",
                        "decision_reason": "test confirmation",
                    },
                    "pending_action": {
                        "action_id": "action_1",
                        "action": "tool",
                        "tool_name": "get_current_time",
                        "tool_arguments": {},
                        "subagent_name": None,
                        "subagent_input": {},
                        "confirmation_hint": "confirm?",
                        "decision_reason": "test confirmation",
                        "requires_confirmation": True,
                        "confirmation_message": "confirm?",
                    },
                    "action_rounds": 1,
                }

        child_builder = StateGraph(_NativePaperSearchState)
        child_builder.add_node("noop", lambda _state: {})
        child_builder.add_edge(START, "noop")
        child_builder.add_edge("noop", END)
        graph = build_main_agent_workflow(
            solve_node=Solve(),
            paper_search_graph=child_builder.compile(),
            checkpointer=InMemorySaver(),
        )
        initial = {
            "conversation_id": "conversation-confirm",
            "run_id": "run-confirm",
            "forced_subagent": None,
            "paper_search_request": None,
            "paper_search_handoff": None,
            "paper_search_action_id": None,
            "messages": [],
            "solve_decision": None,
            "pending_action": None,
            "confirmation": None,
            "last_action_result": None,
            "action_history": [],
            "artifact_refs": [],
            "reply": "",
            "run_status": "running",
            "error": None,
            "action_rounds": 0,
            "max_action_rounds": 8,
        }
        config = {
            "configurable": {
                "thread_id": "conversation-confirm",
            }
        }

        interrupted = await graph.ainvoke(initial, config=config)
        self.assertIn("__interrupt__", interrupted)

        result = await graph.ainvoke(
            Command(resume={"decision": "invalid"}),
            config=config,
        )

        self.assertEqual(result["run_status"], "failed")
        self.assertIn("approved", result["error"])
        self.assertIsNone(result["pending_action"])
        self.assertIsNone(result["solve_decision"])
        self.assertIsNone(result["confirmation"])

    async def test_final_preserves_existing_failure(self) -> None:
        result = await final_node({
            "run_status": "failed",
            "error": "broken state",
            "solve_decision": {
                "action": "direct",
                "answer": "must not complete",
            },
        })

        self.assertEqual(result["run_status"], "failed")
        self.assertEqual(result["error"], "broken state")
        self.assertNotEqual(result["reply"], "must not complete")

    async def test_executes_native_subgraph_and_returns_handoff(self) -> None:
        class Solve:
            calls = 0

            async def __call__(self, _state):
                self.calls += 1
                if self.calls == 1:
                    decision = {
                        "action": "subagent",
                        "answer": None,
                        "tool_name": None,
                        "tool_arguments": {},
                        "subagent_name": "paper_search_agent",
                        "subagent_input": {"prompt": "RAG papers"},
                        "confirmation_hint": None,
                        "decision_reason": "test native subgraph",
                    }
                    return {
                        "solve_decision": decision,
                        "pending_action": {
                            "action_id": "action_1",
                            "action": "subagent",
                            "tool_name": None,
                            "tool_arguments": {},
                            "subagent_name": "paper_search_agent",
                            "subagent_input": {"prompt": "RAG papers"},
                            "confirmation_hint": None,
                            "decision_reason": "test native subgraph",
                            "requires_confirmation": False,
                            "confirmation_message": None,
                        },
                        "action_rounds": 1,
                    }
                return {
                    "solve_decision": {
                        "action": "direct",
                        "answer": "done",
                        "tool_name": None,
                        "tool_arguments": {},
                        "subagent_name": None,
                        "subagent_input": {},
                        "confirmation_hint": None,
                        "decision_reason": "handoff received",
                    },
                    "pending_action": None,
                    "action_rounds": 2,
                }

        async def native_child(state: _NativePaperSearchState):
            self.assertEqual(
                state["paper_search_request"]["prompt"],
                "RAG papers",
            )
            return {
                "paper_search_handoff": {
                    "status": "success",
                    "summary": "found papers",
                    "data": {"workflow_stage": "completed"},
                    "artifact_refs": ["artifact://run/result.json"],
                    "retryable": False,
                    "error_code": None,
                    "error_message": None,
                }
            }

        child_builder = StateGraph(_NativePaperSearchState)
        child_builder.add_node("native_child", native_child)
        child_builder.add_edge(START, "native_child")
        child_builder.add_edge("native_child", END)
        paper_search_graph = child_builder.compile()

        graph = build_main_agent_workflow(
            solve_node=Solve(),
            paper_search_graph=paper_search_graph,
        )
        result = await graph.ainvoke({
            "conversation_id": "conversation-1",
            "run_id": "run-1",
            "forced_subagent": None,
            "paper_search_request": None,
            "paper_search_handoff": None,
            "paper_search_action_id": None,
            "messages": [],
            "solve_decision": None,
            "pending_action": None,
            "confirmation": None,
            "last_action_result": None,
            "action_history": [],
            "artifact_refs": [],
            "reply": "",
            "run_status": "running",
            "error": None,
            "action_rounds": 0,
            "max_action_rounds": 8,
        })

        self.assertEqual(result["reply"], "done")
        self.assertEqual(
            result["last_action_result"]["summary"],
            "found papers",
        )
        self.assertEqual(
            result["artifact_refs"],
            ["artifact://run/result.json"],
        )
        self.assertIsNone(result["paper_search_request"])
        self.assertIsNone(result["paper_search_handoff"])


if __name__ == "__main__":
    unittest.main()
