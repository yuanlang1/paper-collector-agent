import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.llm.graph.main.nodes.solve import SolveNode


class MessageWindowTests(unittest.TestCase):
    def test_keeps_configured_history_turns_and_current_turn(self) -> None:
        messages = [
            HumanMessage(content="turn-1-user"),
            AIMessage(content="turn-1-assistant"),
            HumanMessage(content="turn-2-user"),
            AIMessage(content="turn-2-assistant"),
            HumanMessage(content="turn-3-user"),
            AIMessage(content="", tool_calls=[
                {"name": "example", "args": {}, "id": "call-3"},
            ]),
            ToolMessage(
                name="example",
                tool_call_id="call-3",
                content="tool result",
            ),
            AIMessage(content="turn-3-assistant"),
            HumanMessage(content="turn-4-user"),
        ]

        window = SolveNode._recent_conversation_messages(
            messages,
            history_turns=2,
        )

        self.assertEqual(
            [message.content for message in window],
            [
                "turn-2-user",
                "turn-2-assistant",
                "turn-3-user",
                "",
                "tool result",
                "turn-3-assistant",
                "turn-4-user",
            ],
        )
