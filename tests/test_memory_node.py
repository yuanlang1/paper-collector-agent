import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from app.llm.graph.main.nodes.memory import MemoryNode
from app.llm.graph.main.workflow import build_main_agent_workflow
from app.memory.schemas import MemoryUsage
from app.runtime.system_context import SystemContextResult


class _FakeDb:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeContextBuilder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def build(self, **kwargs) -> SystemContextResult:
        self.calls.append(kwargs)
        callback = kwargs.get("on_memory_event")
        if callback is not None:
            callback(
                {
                    "event": "memory_retrieval_completed",
                    "facts_count": 2,
                    "episodes_count": 1,
                }
            )
        usage: MemoryUsage = {
            "status": "completed",
            "facts_count": 2,
            "episodes_count": 1,
        }
        return SystemContextResult(
            content="dynamic system context",
            memory_usage=usage,
        )


class _CountingMemoryNode:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, _state):
        self.calls += 1
        return {"system_context": "from-memory"}


class _CompletedSolveNode:
    async def __call__(self, state):
        return {
            "messages": [AIMessage(content=state["system_context"])],
            "reply": state["system_context"],
            "pending_tool_calls": [],
            "active_tool_call": None,
            "run_status": "completed",
            "error": None,
        }


class MemoryNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_builds_context_from_current_user_and_emits_memory_event(self):
        db = _FakeDb()
        builder = _FakeContextBuilder()
        node = MemoryNode(
            db_factory=lambda: db,
            system_context_builder=builder,
            llm_config=None,
        )

        with patch("app.llm.graph.main.nodes.memory.emit_custom_event") as emit:
            result = await node(
                {
                    "conversation_id": "conv-1",
                    "user_id": "7",
                    "messages": [
                        HumanMessage(content="earlier"),
                        AIMessage(content="reply"),
                        HumanMessage(content="current request"),
                    ],
                }
            )

        self.assertEqual(result["system_context"], "dynamic system context")
        self.assertEqual(result["memory_usage"]["status"], "completed")
        self.assertTrue(db.closed)
        self.assertEqual(builder.calls[0]["user_id"], "7")
        self.assertEqual(builder.calls[0]["user_message"], "current request")
        emit.assert_called_once_with(
            {
                "event": "memory_retrieval_completed",
                "facts_count": 2,
                "episodes_count": 1,
            }
        )

    async def test_main_workflow_runs_memory_once_before_solve(self):
        memory_node = _CountingMemoryNode()
        graph = build_main_agent_workflow(
            memory_node=memory_node,
            solve_node=_CompletedSolveNode(),
            paper_search_graph=_CompletedSolveNode(),
            task_review_graph=_CompletedSolveNode(),
        )

        result = await graph.ainvoke(
            {
                "conversation_id": "conv-1",
                "user_id": "0",
                "run_id": "run-1",
                "messages": [HumanMessage(content="hello")],
                "pending_tool_calls": [],
                "active_tool_call": None,
                "run_status": "running",
            }
        )

        self.assertEqual(memory_node.calls, 1)
        self.assertEqual(result["reply"], "from-memory")


if __name__ == "__main__":
    unittest.main()
