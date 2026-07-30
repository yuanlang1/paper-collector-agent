import unittest

from app.llm.graph.main.schemas import SolveDecision
from app.llm.streaming.visual_adapter import AgentStreamAdapter


class SubAgentDecisionTests(unittest.TestCase):
    def test_accepts_paper_search_subagent_decision(self) -> None:
        decision = SolveDecision.model_validate({
            "action": "subagent",
            "subagent_name": "paper_search_agent",
            "subagent_input": {"prompt": "RAG papers"},
        })

        self.assertEqual(decision.to_state()["action"], "subagent")

    def test_rejects_removed_workflow_action(self) -> None:
        with self.assertRaises(Exception):
            SolveDecision.model_validate({
                "action": "workflow",
                "workflow_name": "paper_search",
            })

    def test_native_subgraph_update_maps_to_progress_event(self) -> None:
        event, payload = AgentStreamAdapter.subagent_progress(
            subagent="paper_search_agent",
            action_id="action_1",
            child_thread_id=(
                "conversation-1:paper_search_agent:action_1"
            ),
            checkpoint_namespace="paper_search:child-1",
            node_name="search_arxiv",
            update={
                "paper_service_task_id": 42,
                "stage": "normalizing",
                "status": "running",
                "remote_task_state": "SEARCH_RUNNING",
                "progress": {"discovered": 5},
                "warnings": [],
            },
        )

        self.assertEqual(event, "subagent_progress")
        self.assertEqual(payload["delegation_id"], "action_1")
        self.assertEqual(
            payload["checkpoint_namespace"],
            "paper_search:child-1",
        )
        self.assertEqual(payload["node"], "search_arxiv")
        self.assertEqual(payload["progress"]["discovered"], 5)
        self.assertEqual(payload["workflow"], "paper_search")
        self.assertEqual(payload["phase"], "search")
        self.assertEqual(payload["phase_label"], "多来源检索")
        self.assertEqual(payload["task_id"], 42)
        self.assertEqual(
            payload["remote_task_state"],
            "SEARCH_RUNNING",
        )
        self.assertGreater(payload["progress_percent"], 0)

    def test_only_finalize_handoff_is_terminal(self) -> None:
        _, persist = AgentStreamAdapter.subagent_progress(
            subagent="paper_search_agent",
            action_id="action_1",
            child_thread_id="child-1",
            checkpoint_namespace="paper_search:child-1",
            node_name="persist",
            update={
                "stage": "completed",
                "status": "completed",
            },
        )
        _, finalize = AgentStreamAdapter.subagent_progress(
            subagent="paper_search_agent",
            action_id="action_1",
            child_thread_id="child-1",
            checkpoint_namespace="paper_search:child-1",
            node_name="finalize_handoff",
            update={
                "stage": "completed",
                "status": "completed",
            },
        )

        self.assertFalse(persist["terminal"])
        self.assertLess(persist["progress_percent"], 100)
        self.assertTrue(finalize["terminal"])
        self.assertEqual(finalize["progress_percent"], 100)

if __name__ == "__main__":
    unittest.main()
