import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest

from app.history.store import ChatHistoryStore
from app.runtime.agent_runtime import AgentRuntime


COMPLETED_RESPONSE = {
    "conversation_id": "conv_resume",
    "run_id": "run_existing",
    "status": "completed",
    "reply": "Resume completed.",
    "pending_action": None,
    "last_action_result": None,
    "artifact_refs": ["artifact://run_existing/final.json"],
    "interrupt": None,
    "error": None,
}

PENDING_ACTION = {
    "action_id": "call_pending",
    "action_type": "subagent",
    "name": "paper_search_agent",
    "input": {"query": "multimodal keyframes"},
    "requires_confirmation": True,
    "message": "Allow paper_search_agent?",
}

PENDING_CONFIRMATION_RESPONSE = {
    **COMPLETED_RESPONSE,
    "status": "confirmation_required",
    "reply": "Allow paper_search_agent?",
    "pending_action": PENDING_ACTION,
}


class _FakeAgentService:
    def __init__(
        self,
        *,
        response: dict | None = None,
        invoke_error: Exception | None = None,
        stream_terminal: bool = True,
    ) -> None:
        self.response = response or COMPLETED_RESPONSE
        self.invoke_error = invoke_error
        self.stream_terminal = stream_terminal
        self.release_stream = Event()
        self.stream_finished = Event()
        self.stream_sessions = []
        self.stream_loop = None

    async def get_state(self, _session):
        return SimpleNamespace(
            values={
                "run_id": "run_existing",
                "active_tool_call": {
                    "id": "call_pending",
                    "name": "paper_search_agent",
                    "args": {"query": "multimodal keyframes"},
                    "kind": "subagent",
                    "requires_confirmation": True,
                },
            },
            next=("confirm",),
        )

    async def invoke(self, _session):
        if self.invoke_error is not None:
            raise self.invoke_error
        return self.response

    async def stream(self, _session, *, on_terminal=None):
        self.stream_sessions.append(_session)
        self.stream_loop = asyncio.get_running_loop()
        yield "event: run_started\n\n"
        if not self.stream_terminal:
            await asyncio.to_thread(self.release_stream.wait)

        if on_terminal is not None:
            await on_terminal(
                self.response,
                {
                    "schema_version": 1,
                    "card": {
                        "status": self.response["status"],
                        "reasoning": [],
                        "tools": [],
                        "subagents": [],
                    },
                },
            )
        self.stream_finished.set()
        yield "event: run_completed\n\n"


class ResumeHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = TemporaryDirectory()
        self.store = ChatHistoryStore(
            Path(self.temp_dir.name) / "chat-history.db",
        )
        self.store.initialize()

    async def asyncTearDown(self):
        self.temp_dir.cleanup()

    async def _create_pending_confirmation(self) -> int:
        assistant_message_id = await self.store.start_turn(
            conversation_id="conv_resume",
            run_id="run_existing",
            user_content="Find papers about multimodal keyframes.",
        )
        await self.store.complete_assistant_message(
            message_id=assistant_message_id,
            response=PENDING_CONFIRMATION_RESPONSE,
            latency_ms=100,
            extra_meta={
                "schema_version": 1,
                "card": {
                    "status": "confirmation_required",
                    "reasoning": [
                        {
                            "reasoning_id": "run_existing:solve:1",
                            "scope": "main",
                            "text": "I should use paper_search_agent.",
                        }
                    ],
                    "tools": [],
                    "subagents": [
                        {
                            "delegation_id": "call_pending",
                            "name": "paper_search_agent",
                            "status": "awaiting_approval",
                            "timeline": [],
                        }
                    ],
                    "pending_action": PENDING_ACTION,
                    "error": None,
                },
            },
        )
        return assistant_message_id

    async def test_resume_chat_completes_the_existing_confirmation_card(self):
        original_message_id = await self._create_pending_confirmation()
        runtime = AgentRuntime(
            agent_service=_FakeAgentService(),
            history_store=self.store,
        )

        response = await runtime.resume_chat(
            conversation_id="conv_resume",
            resume_payload={
                "decision": "approved",
                "comment": "Continue with the proposed query.",
            },
            db=None,
        )

        items, _ = await self.store.list_messages(
            conversation_id="conv_resume",
            limit=10,
            before_id=None,
        )

        self.assertEqual(response, COMPLETED_RESPONSE)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[1]["id"], original_message_id)
        self.assertEqual(items[1]["role"], "assistant")
        self.assertEqual(items[1]["run_id"], "run_existing")
        self.assertEqual(items[1]["source"], "api")
        self.assertEqual(items[1]["status"], "completed")
        self.assertEqual(items[1]["content"], "Resume completed.")
        self.assertEqual(
            items[1]["meta"]["resume"],
            {
                "decision": "approved",
                "comment": "Continue with the proposed query.",
                "action_id": "call_pending",
                "has_query_understanding_override": False,
                "has_search_tag_override": False,
            },
        )
        self.assertEqual(
            items[1]["meta"]["card"]["approval_history"][0]["decision"],
            "approved",
        )
        self.assertEqual(
            items[1]["meta"]["card"]["subagents"][0]["status"],
            "awaiting_approval",
        )

    async def test_resume_stream_persists_the_terminal_response(self):
        original_message_id = await self._create_pending_confirmation()
        agent_service = _FakeAgentService()
        runtime = AgentRuntime(
            agent_service=agent_service,
            history_store=self.store,
        )
        request_loop = asyncio.get_running_loop()

        events = [
            event
            async for event in runtime.resume_chat_stream(
                conversation_id="conv_resume",
                resume_payload={"decision": "approved"},
                db=None,
            )
        ]
        items, _ = await self.store.list_messages(
            conversation_id="conv_resume",
            limit=10,
            before_id=None,
        )

        self.assertEqual(events[-1], "event: run_completed\n\n")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[1]["id"], original_message_id)
        self.assertEqual(items[1]["status"], "completed")
        self.assertEqual(
            items[1]["meta"]["resume"]["decision"],
            "approved",
        )
        self.assertEqual(items[1]["meta"]["schema_version"], 1)
        self.assertEqual(items[1]["meta"]["card"]["tools"], [])
        self.assertEqual(
            agent_service.stream_sessions[0].assistant_message_id,
            original_message_id,
        )
        self.assertEqual(
            items[1]["meta"]["card"]["subagents"][0]["status"],
            "awaiting_approval",
        )
        self.assertIs(agent_service.stream_loop, request_loop)

    async def test_resume_stream_persists_a_follow_up_confirmation(self):
        original_message_id = await self._create_pending_confirmation()
        response = {
            **COMPLETED_RESPONSE,
            "status": "confirmation_required",
            "reply": "Another confirmation is required.",
        }
        runtime = AgentRuntime(
            agent_service=_FakeAgentService(response=response),
            history_store=self.store,
        )

        _events = [
            event
            async for event in runtime.resume_chat_stream(
                conversation_id="conv_resume",
                resume_payload={"decision": "approved"},
                db=None,
            )
        ]
        items, _ = await self.store.list_messages(
            conversation_id="conv_resume",
            limit=10,
            before_id=None,
        )

        self.assertEqual(len(items), 2)
        self.assertEqual(items[1]["id"], original_message_id)
        self.assertEqual(items[1]["status"], "confirmation_required")
        self.assertEqual(
            items[1]["content"],
            "Another confirmation is required.",
        )

    async def test_detached_resume_stream_continues_to_completion(self):
        original_message_id = await self._create_pending_confirmation()
        agent_service = _FakeAgentService(stream_terminal=False)
        runtime = AgentRuntime(
            agent_service=agent_service,
            history_store=self.store,
        )

        stream = runtime.resume_chat_stream(
            conversation_id="conv_resume",
            resume_payload={"decision": "approved"},
            db=None,
        )
        self.assertEqual(await anext(stream), "event: run_started\n\n")
        await stream.aclose()
        agent_service.release_stream.set()
        await asyncio.wait_for(
            asyncio.to_thread(agent_service.stream_finished.wait),
            timeout=2,
        )

        items, _ = await self.store.list_messages(
            conversation_id="conv_resume",
            limit=10,
            before_id=None,
        )

        self.assertEqual(len(items), 2)
        self.assertEqual(items[1]["id"], original_message_id)
        self.assertEqual(items[1]["status"], "completed")
        self.assertEqual(items[1]["content"], "Resume completed.")
        self.assertEqual(items[1]["meta"]["card"]["status"], "completed")

    async def test_detached_chat_stream_continues_to_completion(self):
        agent_service = _FakeAgentService(stream_terminal=False)
        runtime = AgentRuntime(
            agent_service=agent_service,
            history_store=self.store,
        )

        stream = runtime.chat_stream(
            message="Find papers about multimodal keyframes.",
            conversation_id="conv_chat",
            db=None,
        )
        self.assertEqual(await anext(stream), "event: run_started\n\n")
        await stream.aclose()
        agent_service.release_stream.set()
        await asyncio.wait_for(
            asyncio.to_thread(agent_service.stream_finished.wait),
            timeout=2,
        )

        items, _ = await self.store.list_messages(
            conversation_id="conv_chat",
            limit=10,
            before_id=None,
        )

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["role"], "user")
        self.assertEqual(items[1]["role"], "assistant")
        self.assertEqual(items[1]["status"], "completed")
        self.assertEqual(items[1]["meta"]["card"]["status"], "completed")
        self.assertIsNotNone(agent_service.stream_sessions[0].db)
        self.assertEqual(agent_service.stream_sessions[0].assistant_message_id, items[1]["id"])

    async def test_resume_error_marks_the_placeholder_interrupted(self):
        original_message_id = await self._create_pending_confirmation()
        runtime = AgentRuntime(
            agent_service=_FakeAgentService(
                invoke_error=RuntimeError("resume failed"),
            ),
            history_store=self.store,
        )

        with self.assertRaisesRegex(RuntimeError, "resume failed"):
            await runtime.resume_chat(
                conversation_id="conv_resume",
                resume_payload={"decision": "approved"},
                db=None,
            )

        items, _ = await self.store.list_messages(
            conversation_id="conv_resume",
            limit=10,
            before_id=None,
        )

        self.assertEqual(len(items), 2)
        self.assertEqual(items[1]["id"], original_message_id)
        self.assertEqual(items[1]["status"], "interrupted")

    async def test_resume_rejects_a_different_action_without_changing_history(self):
        original_message_id = await self._create_pending_confirmation()
        runtime = AgentRuntime(
            agent_service=_FakeAgentService(),
            history_store=self.store,
        )

        with self.assertRaisesRegex(ValueError, "does not match"):
            await runtime.resume_chat(
                conversation_id="conv_resume",
                resume_payload={"decision": "approved"},
                requested_action_id="call_other",
                db=None,
            )

        items, _ = await self.store.list_messages(
            conversation_id="conv_resume",
            limit=10,
            before_id=None,
        )
        self.assertEqual(len(items), 2)
        self.assertEqual(items[1]["id"], original_message_id)
        self.assertEqual(items[1]["status"], "confirmation_required")


if __name__ == "__main__":
    unittest.main()
