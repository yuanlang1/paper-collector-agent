import unittest
from unittest import mock

from app.llm.streaming.card_snapshot import CardMetaAccumulator


class CardMetaAccumulatorTests(unittest.TestCase):
    def test_confirmation_action_is_marked_as_awaiting_approval(self):
        accumulator = CardMetaAccumulator()

        self._observe(
            accumulator,
            1,
            "action_started",
            {
                "action_id": "delegation_1",
                "action_type": "subagent",
                "name": "paper_search_agent",
                "requires_confirmation": True,
            },
        )

        meta = accumulator.snapshot(
            {
                "status": "confirmation_required",
                "pending_action": {"action_id": "delegation_1"},
            }
        )

        self.assertEqual(
            meta["card"]["subagents"][0]["status"],
            "awaiting_approval",
        )

    def test_snapshot_keeps_tools_and_subagents_separate(self):
        accumulator = CardMetaAccumulator(
            model="test-model",
            provider="test-provider",
        )

        self._observe(
            accumulator,
            1,
            "iteration_started",
            {"iteration": 2},
        )

        self._observe(
            accumulator,
            2,
            "reasoning_delta",
            {
                "reasoning_id": "run_1:solve:1",
                "scope": "main",
                "delta": "先分析需求。",
            },
        )
        self._observe(
            accumulator,
            3,
            "reasoning_delta",
            {
                "reasoning_id": "run_1:solve:1",
                "scope": "main",
                "delta": "再调用工作流。",
            },
        )
        self._observe(
            accumulator,
            4,
            "action_started",
            {
                "action_id": "delegation_1",
                "action_type": "subagent",
                "name": "paper_search_agent",
                "workflow": "paper_search",
                "input": {"query": "graph retrieval"},
            },
        )
        self._observe(
            accumulator,
            5,
            "timeline_step",
            {
                "delegation_id": "delegation_1",
                "workflow": "paper_search",
                "step_id": "paper_search:query_plan",
                "step_key": "query_plan",
                "label": "生成检索计划",
                "state": "started",
            },
        )
        self._observe(
            accumulator,
            6,
            "timeline_step",
            {
                "delegation_id": "delegation_1",
                "workflow": "paper_search",
                "step_id": "paper_search:query_plan",
                "step_key": "query_plan",
                "label": "生成检索计划",
                "state": "completed",
            },
        )
        self._observe(
            accumulator,
            7,
            "action_result",
            {
                "action_id": "delegation_1",
                "action_type": "subagent",
                "name": "paper_search_agent",
                "status": "success",
                "summary": "workflow completed",
                "artifact_refs": ["artifact://search/1"],
            },
        )
        self._observe(
            accumulator,
            8,
            "action_started",
            {
                "action_id": "tool_1",
                "action_type": "tool",
                "name": "get_task",
                "input": {"task_id": "task_1"},
            },
        )
        self._observe(
            accumulator,
            9,
            "action_result",
            {
                "action_id": "tool_1",
                "action_type": "tool",
                "name": "get_task",
                "status": "success",
                "summary": "tool completed",
            },
        )

        meta = accumulator.snapshot(
            {
                "status": "completed",
                "artifact_refs": ["artifact://search/1"],
                "pending_action": None,
                "error": None,
            }
        )

        self.assertEqual(meta["schema_version"], 1)
        self.assertEqual(meta["card"]["iterations"], 2)
        self.assertEqual(meta["card"]["model"], "test-model")
        self.assertEqual(meta["card"]["provider"], "test-provider")
        self.assertEqual(
            meta["card"]["reasoning"],
            [
                {
                    "reasoning_id": "run_1:solve:1",
                    "scope": "main",
                    "start_seq": 2,
                    "end_seq": 3,
                    "started_at": "2026-08-30T00:00:02+00:00",
                    "finished_at": "2026-08-30T00:00:03+00:00",
                    "text": "先分析需求。再调用工作流。",
                }
            ],
        )
        self.assertEqual(
            meta["card"]["tools"],
            [
                {
                    "action_id": "tool_1",
                    "name": "get_task",
                    "status": "success",
                    "start_seq": 8,
                    "end_seq": 9,
                    "started_at": "2026-08-30T00:00:08+00:00",
                    "finished_at": "2026-08-30T00:00:09+00:00",
                    "duration_ms": mock.ANY,
                    "input": {"task_id": "task_1"},
                    "summary": "tool completed",
                    "artifact_refs": [],
                    "retryable": False,
                    "error_code": None,
                    "error_message": None,
                }
            ],
        )
        subagent = meta["card"]["subagents"][0]
        self.assertEqual(subagent["delegation_id"], "delegation_1")
        self.assertEqual(subagent["workflow"], "paper_search")
        self.assertEqual(subagent["start_seq"], 4)
        self.assertEqual(subagent["end_seq"], 7)
        self.assertEqual(subagent["status"], "success")
        self.assertEqual(
            subagent["timeline"],
            [
                {
                    "step_id": "paper_search:query_plan",
                    "step_key": "query_plan",
                    "label": "生成检索计划",
                    "iteration": None,
                    "state": "completed",
                    "start_seq": 5,
                    "end_seq": 6,
                    "started_at": "2026-08-30T00:00:05+00:00",
                    "finished_at": "2026-08-30T00:00:06+00:00",
                    "duration_ms": mock.ANY,
                    "error": None,
                }
            ],
        )

    def test_snapshot_records_memory_retrieval_status_and_counts(self):
        accumulator = CardMetaAccumulator()

        self._observe(
            accumulator,
            1,
            "memory_retrieval_started",
            {"facts_count": 0, "episodes_count": 0},
        )
        self._observe(
            accumulator,
            2,
            "memory_retrieval_completed",
            {"facts_count": 3, "episodes_count": 1},
        )

        memory = accumulator.snapshot({"status": "completed"})["card"]["memory"]

        self.assertEqual(
            memory,
            {
                "status": "completed",
                "facts_count": 3,
                "episodes_count": 1,
                "start_seq": 1,
                "started_at": "2026-08-30T00:00:01+00:00",
                "end_seq": 2,
                "finished_at": "2026-08-30T00:00:02+00:00",
            },
        )

    @staticmethod
    def _observe(
        accumulator: CardMetaAccumulator,
        sequence: int,
        event: str,
        data: dict,
    ) -> None:
        accumulator.observe(
            {
                "sequence": sequence,
                "event": event,
                "timestamp": f"2026-08-30T00:00:{sequence:02d}+00:00",
                "data": data,
            }
        )


if __name__ == "__main__":
    unittest.main()
