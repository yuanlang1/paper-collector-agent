from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.llm.graph.main.nodes.memory import MemoryNode
from app.llm.graph.main.nodes.solve import SolveNode
from app.memory.soul import SoulLoadError, load_soul
from app.runtime.system_context import SystemContextBuilder


class _NoMatchingSkills:
    def matching_instructions(self, _message: str) -> str:
        return ""


class _FakeDb:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _SoulFailingContextBuilder:
    async def build(self, **_kwargs):
        raise SoulLoadError("soul is unavailable")


class _RecordingModel:
    def __init__(self) -> None:
        self.messages = []

    async def astream(self, messages):
        self.messages = messages
        yield AIMessage(content="done")


class SoulContextTests(unittest.IsolatedAsyncioTestCase):
    def test_load_soul_reads_the_packaged_markdown(self) -> None:
        expected = (
            Path(__file__).parents[1]
            / "app"
            / "memory"
            / "soul"
            / "soul.md"
        ).read_text(encoding="utf-8").strip()

        self.assertEqual(load_soul(), expected)

    def test_load_soul_rejects_empty_content(self) -> None:
        with TemporaryDirectory() as temp_dir:
            empty_path = Path(temp_dir) / "soul.md"
            empty_path.write_text("\n", encoding="utf-8")

            with patch("app.memory.soul.SOUL_PATH", empty_path):
                with self.assertRaisesRegex(SoulLoadError, "empty"):
                    load_soul()

    async def test_dynamic_context_starts_with_soul(self) -> None:
        builder = SystemContextBuilder()
        builder.skill_loader = _NoMatchingSkills()

        with patch(
            "app.runtime.system_context.load_soul",
            return_value="Soul rules",
        ):
            result = await builder.build(
                user_message="hello",
                user_id="0",
                conversation_id="conv_1",
                db=None,
                llm_config=None,
            )

        self.assertTrue(result.content.startswith("Soul rules\n\n## Current time"))

    async def test_memory_node_reraises_soul_load_error(self) -> None:
        db = _FakeDb()
        node = MemoryNode(
            db_factory=lambda: db,
            system_context_builder=_SoulFailingContextBuilder(),
            llm_config=None,
        )

        with self.assertRaises(SoulLoadError):
            await node(
                {
                    "conversation_id": "conv_1",
                    "user_id": "0",
                    "messages": [HumanMessage(content="hello")],
                }
            )

        self.assertTrue(db.closed)

    async def test_solve_uses_only_dynamic_system_context(self) -> None:
        model = _RecordingModel()
        node = SolveNode(model=model)

        result = await node(
            {
                "run_id": "run_1",
                "system_context": "Soul rules\n\nDynamic context",
                "messages": [HumanMessage(content="hello")],
                "iteration_count": 0,
            }
        )

        self.assertEqual(result["reply"], "done")
        self.assertEqual(len(model.messages), 2)
        self.assertIsInstance(model.messages[0], SystemMessage)
        self.assertEqual(model.messages[0].content, "Soul rules\n\nDynamic context")
        self.assertIsInstance(model.messages[1], HumanMessage)

    async def test_solve_uses_the_conversation_window_anchor(self) -> None:
        model = _RecordingModel()
        node = SolveNode(model=model)

        await node(
            {
                "run_id": "run_1",
                "system_context": "Soul rules",
                "conversation_window_start_id": "current",
                "messages": [
                    HumanMessage(id="older", content="older request"),
                    AIMessage(id="older-reply", content="older reply"),
                    HumanMessage(id="current", content="current request"),
                ],
                "iteration_count": 0,
            }
        )

        self.assertEqual(
            [message.content for message in model.messages],
            ["Soul rules", "current request"],
        )


if __name__ == "__main__":
    unittest.main()
