import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
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

    async def get_state(self, _session):
        return SimpleNamespace(
            values={"run_id": "run_existing"},
            next=("confirm",),
        )

    async def invoke(self, _session):
        if self.invoke_error is not None:
            raise self.invoke_error
        return self.response

    async def stream(self, _session, *, on_terminal=None):
        yield "event: run_started\n\n"
        if not self.stream_terminal:
            await asyncio.Event().wait()

        if on_terminal is not None:
            await on_terminal(self.response)
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

    async def test_resume_chat_appends_a_completed_assistant_message(self):
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
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["role"], "assistant")
        self.assertEqual(items[0]["run_id"], "run_existing")
        self.assertEqual(items[0]["source"], "resume")
        self.assertEqual(items[0]["status"], "completed")
        self.assertEqual(items[0]["content"], "Resume completed.")
        self.assertEqual(
            items[0]["meta"]["resume"],
            {
                "decision": "approved",
                "comment": "Continue with the proposed query.",
                "has_query_understanding_override": False,
                "has_search_tag_override": False,
            },
        )

    async def test_resume_stream_persists_the_terminal_response(self):
        runtime = AgentRuntime(
            agent_service=_FakeAgentService(),
            history_store=self.store,
        )

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
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["status"], "completed")
        self.assertEqual(
            items[0]["meta"]["resume"]["decision"],
            "approved",
        )

    async def test_resume_stream_persists_a_follow_up_confirmation(self):
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

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["status"], "confirmation_required")
        self.assertEqual(
            items[0]["content"],
            "Another confirmation is required.",
        )

    async def test_interrupted_resume_stream_marks_the_placeholder(self):
        runtime = AgentRuntime(
            agent_service=_FakeAgentService(stream_terminal=False),
            history_store=self.store,
        )

        stream = runtime.resume_chat_stream(
            conversation_id="conv_resume",
            resume_payload={"decision": "approved"},
            db=None,
        )
        self.assertEqual(await anext(stream), "event: run_started\n\n")
        await stream.aclose()

        items, _ = await self.store.list_messages(
            conversation_id="conv_resume",
            limit=10,
            before_id=None,
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["status"], "interrupted")

    async def test_resume_error_marks_the_placeholder_interrupted(self):
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

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
