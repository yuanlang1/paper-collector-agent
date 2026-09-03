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

    def test_keeps_tool_messages_added_after_window_anchor(self) -> None:
        messages = [
            HumanMessage(id="turn-1-user", content="turn-1-user"),
            AIMessage(id="turn-1-assistant", content="turn-1-assistant"),
            HumanMessage(id="turn-2-user", content="turn-2-user"),
            AIMessage(
                id="turn-2-tool-call",
                content="",
                tool_calls=[
                    {"name": "example", "args": {}, "id": "call-2"},
                ],
            ),
            ToolMessage(
                id="turn-2-tool-result",
                name="example",
                tool_call_id="call-2",
                content="tool result",
            ),
        ]

        window = SolveNode._messages_from_window_start(
            messages,
            conversation_window_start_id="turn-2-user",
        )

        self.assertEqual(
            [message.content for message in window],
            ["turn-2-user", "", "tool result"],
        )
