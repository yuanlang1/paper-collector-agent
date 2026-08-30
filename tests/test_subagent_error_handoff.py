import json
import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphInterrupt

from app.llm.graph.main.nodes.paper_search import complete_paper_search_node
from app.llm.graph.main.nodes.safe_subgraph import SafeSubgraphNode
from app.llm.graph.main.nodes.task_review import complete_task_review_node
from app.llm.graph.main.workflow import build_main_agent_workflow


class _FailingSubgraph:
    async def ainvoke(self, _state, config):
        del config
        raise TypeError("subgraph configuration is invalid")


class _InterruptedSubgraph:
    async def ainvoke(self, _state, config):
        del config
        raise GraphInterrupt()


class _ReplyAfterSubagentFailure:
    def __init__(self, *, tool_name: str) -> None:
        self.tool_name = tool_name

    async def __call__(self, state):
        if any(isinstance(message, ToolMessage) for message in state["messages"]):
            return {
                "messages": [AIMessage(content="subagent failure received")],
                "reply": "subagent failure received",
                "pending_tool_calls": [],
                "active_tool_call": None,
                "run_status": "completed",
                "error": None,
            }

        call = {
            "id": "call_subagent",
            "name": self.tool_name,
            "args": (
                {"prompt": "AQA"}
                if self.tool_name == "paper_search_agent"
                else {"task_id": 1, "topic": "AQA"}
            ),
        }
        return {
            "messages": [AIMessage(content="", tool_calls=[call])],
            "pending_tool_calls": [
                {
                    **call,
                    "kind": "subagent",
                    "requires_confirmation": False,
                }
            ],
            "active_tool_call": None,
            "run_status": "running",
        }


class SubagentErrorHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_main_workflow_returns_to_solve_after_paper_search_failure(self):
        result = await self._run_main_workflow("paper_search_agent")

        tool_messages = [
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        ]

        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].tool_call_id, "call_subagent")
        self.assertEqual(
            json.loads(tool_messages[0].content)["error_code"],
            "PAPER_SEARCH_SUBGRAPH_FAILED",
        )
        self.assertEqual(result["reply"], "subagent failure received")

    async def test_main_workflow_returns_to_solve_after_task_review_failure(self):
        result = await self._run_main_workflow("task_review_agent")

        tool_messages = [
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        ]

        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].tool_call_id, "call_subagent")
        self.assertEqual(
            json.loads(tool_messages[0].content)["error_code"],
            "TASK_REVIEW_SUBGRAPH_FAILED",
        )
        self.assertEqual(result["reply"], "subagent failure received")

    async def test_paper_search_exception_becomes_a_matching_tool_message(self):
        handoff = await self._failed_handoff(
            handoff_key="paper_search_handoff",
            subagent="paper_search_agent",
            error_code="PAPER_SEARCH_SUBGRAPH_FAILED",
        )
        result = await complete_paper_search_node(
            {
                "paper_search_handoff": handoff,
                "paper_search_tool_call_id": "call_subagent",
            }
        )

        tool_message = result["messages"][0]
        payload = json.loads(tool_message.content)

        self.assertEqual(tool_message.tool_call_id, "call_subagent")
        self.assertEqual(payload["error_code"], "PAPER_SEARCH_SUBGRAPH_FAILED")
        self.assertEqual(payload["status"], "error")

    async def test_task_review_exception_becomes_a_matching_tool_message(self):
        handoff = await self._failed_handoff(
            handoff_key="task_review_handoff",
            subagent="task_review_agent",
            error_code="TASK_REVIEW_SUBGRAPH_FAILED",
        )
        result = await complete_task_review_node(
            {
                "task_review_handoff": handoff,
                "task_review_tool_call_id": "call_subagent",
            }
        )

        tool_message = result["messages"][0]
        payload = json.loads(tool_message.content)

        self.assertEqual(tool_message.tool_call_id, "call_subagent")
        self.assertEqual(payload["error_code"], "TASK_REVIEW_SUBGRAPH_FAILED")
        self.assertEqual(payload["status"], "error")

    async def test_invalid_paper_search_handoff_still_returns_a_tool_message(self):
        result = await complete_paper_search_node(
            {
                "paper_search_handoff": {"status": "error"},
                "paper_search_tool_call_id": "call_subagent",
            }
        )

        payload = json.loads(result["messages"][0].content)

        self.assertEqual(payload["error_code"], "PAPER_SEARCH_HANDOFF_INVALID")

    async def test_invalid_task_review_handoff_still_returns_a_tool_message(self):
        result = await complete_task_review_node(
            {
                "task_review_handoff": {"status": "error"},
                "task_review_tool_call_id": "call_subagent",
            }
        )

        payload = json.loads(result["messages"][0].content)

        self.assertEqual(payload["error_code"], "TASK_REVIEW_HANDOFF_INVALID")

    async def test_interrupt_is_not_converted_to_a_failure_handoff(self):
        node = SafeSubgraphNode(
            subgraph=_InterruptedSubgraph(),
            handoff_key="paper_search_handoff",
            subagent="paper_search_agent",
            error_code="PAPER_SEARCH_SUBGRAPH_FAILED",
            summary="subagent failed",
        )

        with self.assertRaises(GraphInterrupt):
            await node({}, {})

    async def _failed_handoff(
        self,
        *,
        handoff_key: str,
        subagent: str,
        error_code: str,
    ):
        node = SafeSubgraphNode(
            subgraph=_FailingSubgraph(),
            handoff_key=handoff_key,
            subagent=subagent,
            error_code=error_code,
            summary="subagent failed",
        )
        with self.assertLogs(
            "app.llm.graph.main.nodes.safe_subgraph",
            level="ERROR",
        ):
            result = await node({}, {})

        self.assertIn(handoff_key, result)
        return result[handoff_key]

    async def _run_main_workflow(self, tool_name: str):
        graph = build_main_agent_workflow(
            solve_node=_ReplyAfterSubagentFailure(tool_name=tool_name),
            paper_search_graph=_FailingSubgraph(),
            task_review_graph=_FailingSubgraph(),
        )
        with patch.dict(
            os.environ,
            {"LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false"},
        ):
            with self.assertLogs(
                "app.llm.graph.main.nodes.safe_subgraph",
                level="ERROR",
            ):
                return await graph.ainvoke(
                    {
                        "conversation_id": "conversation-1",
                        "run_id": "run-1",
                        "messages": [HumanMessage(content="run the subagent")],
                        "pending_tool_calls": [],
                        "active_tool_call": None,
                        "run_status": "running",
                    }
                )


if __name__ == "__main__":
    unittest.main()
