import asyncio
from tempfile import TemporaryDirectory
import unittest

from app.history.store import ChatHistoryStore


class MemoryHistoryIntegrationTests(unittest.TestCase):
    def test_completed_turns_are_scoped_by_default_user(self) -> None:
        async def scenario(db_path: str) -> None:
            store = ChatHistoryStore(db_path)
            store.initialize()

            message_id = await store.start_turn(
                conversation_id="conversation-1",
                run_id="run-1",
                user_content="I only want open-access papers.",
            )
            await store.complete_assistant_message(
                message_id=message_id,
                response={"status": "completed", "reply": "I will filter for OA."},
                latency_ms=10,
            )

            turns = await store.list_completed_turns_after(
                user_id="0",
                conversation_id="conversation-1",
                after_assistant_message_id=None,
                limit=6,
            )
            self.assertEqual(
                [(turn.user_content, turn.assistant_content) for turn in turns],
                [("I only want open-access papers.", "I will filter for OA.")],
            )

            other_user_turns = await store.list_completed_turns_after(
                user_id="other-user",
                conversation_id="conversation-1",
                after_assistant_message_id=None,
                limit=6,
            )
            self.assertEqual(other_user_turns, [])

        with TemporaryDirectory() as temp_dir:
            asyncio.run(scenario(f"{temp_dir}/history.db"))
