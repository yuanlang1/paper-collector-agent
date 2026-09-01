import asyncio
from types import SimpleNamespace
import unittest

from app.llm.agent import AgentService
from app.runtime.session import Session


class _FakeGraph:
    async def astream(self, *_args, **_kwargs):
        yield (
            (),
            "updates",
            {
                "dispatch": {
                    "active_tool_call": {
                        "id": "delegation_1",
                        "name": "paper_search_agent",
                        "args": {"query": "graph retrieval"},
                        "kind": "subagent",
                        "requires_confirmation": False,
                    }
                }
            },
        )
        yield (
            (),
            "updates",
            {
                "prepare_paper_search": {
                    "paper_search_tool_call_id": "delegation_1",
                }
            },
        )
        yield (
            ("paper_search:child",),
            "custom",
            {
                "event": "timeline_step",
                "workflow": "paper_search",
                "step_id": "paper_search:query_plan",
                "step_key": "query_plan",
                "label": "生成检索计划",
                "state": "started",
            },
        )
        yield (
            ("paper_search:child",),
            "custom",
            {
                "event": "timeline_step",
                "workflow": "paper_search",
                "step_id": "paper_search:query_plan",
                "step_key": "query_plan",
                "label": "生成检索计划",
                "state": "completed",
            },
        )
        yield (
            (),
            "updates",
            {
                "complete_paper_search": {
                    "last_action_result": {
                        "action_id": "delegation_1",
                        "action_type": "subagent",
                        "name": "paper_search_agent",
                        "status": "success",
                        "summary": "workflow completed",
                        "artifact_refs": [],
                        "retryable": False,
                        "error_code": None,
                        "error_message": None,
                    }
                }
            },
        )

    async def aget_state(self, _config):
        return SimpleNamespace(
            values={
                "run_id": "run_1",
                "run_status": "completed",
                "reply": "Search completed.",
                "reasoning_content": "",
                "active_tool_call": None,
                "last_action_result": None,
                "artifact_refs": [],
                "error": None,
            }
        )


class AgentStreamHistoryMetaTests(unittest.TestCase):
    def test_terminal_callback_receives_separated_execution_card(self):
        service = AgentService.__new__(AgentService)
        service.graph = _FakeGraph()
        session = Session.create(
            message="Find papers about graph retrieval.",
            conversation_id="conv_1",
            db=None,
        )
        captured: list[tuple[dict, dict]] = []

        async def persist_terminal(response: dict, card_meta: dict) -> None:
            captured.append((response, card_meta))

        events = asyncio.run(
            self._collect_events(
                service,
                session,
                persist_terminal,
            )
        )

        self.assertTrue(any("event: action_started" in event for event in events))
        self.assertTrue(any("event: timeline_step" in event for event in events))
        self.assertEqual(len(captured), 1)
        response, meta = captured[0]
        self.assertEqual(response["status"], "completed")
        self.assertEqual(meta["schema_version"], 1)
        self.assertEqual(meta["card"]["tools"], [])
        self.assertEqual(len(meta["card"]["subagents"]), 1)
        self.assertEqual(
            meta["card"]["subagents"][0]["delegation_id"],
            "delegation_1",
        )
        self.assertEqual(
            meta["card"]["subagents"][0]["timeline"][0]["state"],
            "completed",
        )

    @staticmethod
    async def _collect_events(
        service: AgentService,
        session: Session,
        persist_terminal,
    ) -> list[str]:
        return [
            event
            async for event in service.stream(
                session,
                on_terminal=persist_terminal,
            )
        ]


if __name__ == "__main__":
    unittest.main()
