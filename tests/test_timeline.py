import unittest
from unittest.mock import patch

from app.llm.streaming.timeline import (
    PAPER_SEARCH_TIMELINE,
    TASK_REVIEW_TIMELINE,
    _event_payload,
    _step_for_node,
    instrument_timeline_node,
)
from app.llm.streaming.visual_adapter import AgentStreamAdapter


class TimelinePayloadTests(unittest.TestCase):
    def test_paper_search_repeated_step_uses_supplemental_round(self):
        step = _step_for_node(PAPER_SEARCH_TIMELINE, "search_arxiv")
        assert step is not None

        payload = _event_payload(
            workflow="paper_search",
            step=step,
            state={"supplemental_search_round": 1},
            event_state="started",
        )

        self.assertEqual(
            payload["step_id"],
            "paper_search:multi_source_search:2",
        )
        self.assertEqual(payload["label"], "多来源检索（第 2 轮）")
        self.assertEqual(payload["iteration"], 2)

    def test_review_repeated_step_uses_reflection_round(self):
        step = _step_for_node(TASK_REVIEW_TIMELINE, "render_sections")
        assert step is not None

        payload = _event_payload(
            workflow="task_review",
            step=step,
            state={"reflection_round": 1},
            event_state="started",
        )

        self.assertEqual(
            payload["step_id"],
            "task_review:render_sections:2",
        )
        self.assertEqual(payload["label"], "撰写或重写章节（第 2 轮）")

    def test_subagent_progress_exposes_workflow_iteration(self):
        _event, payload = AgentStreamAdapter.subagent_progress(
            subagent="task_review_agent",
            action_id="action_1",
            child_thread_id="child_1",
            checkpoint_namespace="task_review:1",
            node_name="reflect_review",
            update={"reflection_round": 2},
            workflow="task_review",
        )

        self.assertEqual(payload["iteration"], 3)

    def test_wrapper_emits_a_started_and_completed_event(self):
        async def node(_state):
            return {}

        wrapped = instrument_timeline_node(
            workflow="paper_search",
            node_name="generate_queries",
            node=node,
            timeline=PAPER_SEARCH_TIMELINE,
        )

        with patch("app.llm.streaming.timeline.emit_custom_event") as emit:
            import asyncio

            asyncio.run(wrapped({}, {}))

        self.assertEqual(
            [call.args[0]["state"] for call in emit.call_args_list],
            ["started", "completed"],
        )

    def test_wrapper_emits_a_failed_event_for_a_node_error(self):
        async def node(_state):
            return {"error": "upstream unavailable"}

        wrapped = instrument_timeline_node(
            workflow="task_review",
            node_name="generate_claims",
            node=node,
            timeline=TASK_REVIEW_TIMELINE,
        )

        with patch("app.llm.streaming.timeline.emit_custom_event") as emit:
            import asyncio

            asyncio.run(wrapped({"reflection_round": 0}, {}))

        self.assertEqual(
            [call.args[0]["state"] for call in emit.call_args_list],
            ["started", "failed"],
        )

    def test_wrapper_forwards_config_to_nodes_that_require_it(self):
        received_configs = []

        async def node(_state, config):
            received_configs.append(config)
            return {}

        wrapped = instrument_timeline_node(
            workflow="paper_search",
            node_name="search_arxiv",
            node=node,
            timeline=PAPER_SEARCH_TIMELINE,
        )
        config = {"configurable": {"run_id": "run-1"}}

        import asyncio

        asyncio.run(wrapped({}, config))

        self.assertEqual(received_configs, [config])


if __name__ == "__main__":
    unittest.main()
