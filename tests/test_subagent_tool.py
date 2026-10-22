import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import tracing_context

from app.llm.agent_build import build_tools
from app.llm.agent_runner import AgentRunner
from app.llm.context import BaseAgentContext
from app.llm.loop.agent_loop import AgentLoop
from app.llm.tools.agent_tools.subagent import build_subagent_tool, build_subagent_tools
from app.llm.tools.registry import ToolRegistry
from tests.test_agent_compact import RecordingModel
from tests.test_tool_node import invoke_tool, tool_graph


TARGET = "app.llm.tools.agent_tools.subagent"


class SubagentToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(tracing_context(enabled=False))

    def test_registration_exposes_only_description_and_prompt(self):
        model = Mock()
        selected = next(selected for selected in build_tools(None, user_id=1, model=model).get_tools() if selected.name == "subagent")
        schema = convert_to_openai_tool(selected)["function"]["parameters"]
        self.assertEqual(set(schema["properties"]), {"description", "prompt"})
        self.assertEqual(set(schema["required"]), {"description", "prompt"})
        self.assertNotIn("requires_confirmation", selected.metadata or {})
        self.assertEqual({selected.name for selected in build_subagent_tools()}, {"search_web"})

    async def test_invalid_inputs_fail_before_constructing_a_child_graph(self):
        model = Mock()
        selected = build_subagent_tool(model=model)
        for args in (
            {"prompt": "task"}, {"description": "task"},
            {"description": " ", "prompt": "task"},
            {"description": "task", "prompt": "\n"},
            {"description": "task", "prompt": "task", "tools": []},
        ):
            with self.subTest(args=args):
                result = await invoke_tool(selected, args)
                self.assertEqual(result.status, "error")
        model.bind_tools.assert_not_called()

    async def test_child_uses_internal_tools_and_a_fresh_context(self):
        observed = []

        @tool
        async def inspect_child(runtime: ToolRuntime[BaseAgentContext]) -> str:
            """Inspect the isolated child context."""
            observed.append(runtime)
            return "inspected"

        model = RecordingModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "inspect_child", "args": {}, "id": "child-call"}]),
            AIMessage(content="child result"),
        ])
        parent = BaseAgentContext(
            thread_id="parent-thread", run_id="parent-run", user_id="owner",
            auto_threshold=0.7, context_window_tokens=1_000, keep_last_n_turns=1,
        )
        with patch(TARGET + ".search_web_tool", inspect_child):
            result = await invoke_tool(build_subagent_tool(model=model), {
                "description": " Inspect ", "prompt": " Complete this task ",
            }, context=parent)
        self.assertEqual((result.status, result.tool_call_id, result.content), ("success", "call-test", "child result"))
        runtime = observed[0]
        self.assertEqual(runtime.context.user_id, parent.user_id)
        self.assertNotEqual(runtime.context.thread_id, parent.thread_id)
        self.assertNotEqual(runtime.context.run_id, parent.run_id)
        self.assertEqual(runtime.context.auto_threshold, parent.auto_threshold)
        self.assertEqual(runtime.context.context_window_tokens, parent.context_window_tokens)
        self.assertEqual(runtime.config["configurable"]["thread_id"], runtime.context.thread_id)
        self.assertEqual(runtime.config["metadata"]["app_run_id"], runtime.context.run_id)
        self.assertEqual(runtime.config["metadata"]["parent_run_id"], parent.run_id)
        self.assertEqual(runtime.config["metadata"]["parent_thread_id"], parent.thread_id)
        self.assertEqual(runtime.config["metadata"]["parent_tool_call_id"], runtime.tool_call_id)
        self.assertEqual(runtime.config["metadata"]["agent_scope"], "subagent")
        self.assertEqual([message.content for message in model.calls[0]["messages"] if isinstance(message, HumanMessage)], ["Complete this task"])

    async def test_parallel_calls_do_not_share_state_or_identity(self):
        model = RecordingModel(responses=[AIMessage(content="child result")])
        selected = build_subagent_tool(model=model)
        parent = BaseAgentContext(thread_id="parent", run_id="parent-run", user_id="owner")
        with patch(TARGET + ".AgentLoop", wraps=AgentLoop) as constructor:
            result = await tool_graph([selected]).ainvoke({"messages": [AIMessage(content="", tool_calls=[
                {"name": "subagent", "args": {"description": "first", "prompt": "first task"}, "id": "first-call"},
                {"name": "subagent", "args": {"description": "second", "prompt": "second task"}, "id": "second-call"},
            ])]}, context=parent)
        contexts = [call.kwargs["context"] for call in constructor.call_args_list]
        self.assertEqual(len({context.thread_id for context in contexts}), 2)
        self.assertEqual(len({context.run_id for context in contexts}), 2)
        self.assertTrue(all(context.user_id == parent.user_id for context in contexts))
        self.assertTrue(all(call.kwargs["checkpointer"] is False for call in constructor.call_args_list))
        self.assertEqual([message.tool_call_id for message in result["messages"][1:]], ["first-call", "second-call"])
        self.assertEqual(len(result["messages"]), 3)
        self.assertEqual(sorted(
            [message.content for message in call["messages"] if isinstance(message, HumanMessage)]
            for call in model.calls
        ), [["first task"], ["second task"]])

    async def test_parent_stream_includes_child_execution_without_child_checkpoints(self):
        @tool
        async def lookup() -> str:
            """Return a local child result."""
            return "child lookup"

        model = RecordingModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "subagent", "args": {"description": "Inspect sources", "prompt": "child task"}, "id": "delegate-call",
            }]),
            AIMessage(content="", tool_calls=[{"name": "lookup", "args": {}, "id": "lookup-call"}]),
            AIMessage(content="child final"),
            AIMessage(content="parent final"),
        ])
        saver = InMemorySaver()
        registry = ToolRegistry()
        registry.register(build_subagent_tool(model=model), "子代理")
        runner = AgentRunner(
            context=BaseAgentContext(thread_id="parent", run_id="parent-run", user_id="owner"),
            model=model, tool_registry=registry, checkpointer=saver, store=None,
        )
        with patch(TARGET + ".search_web_tool", lookup):
            events = [event async for event in runner.run({"message": "parent task"})]
        tokens = [event for event in events if event["type"] == "token"]
        self.assertEqual(tokens, [{"type": "token", "content": "child final", "parent_tool_call_id": "delegate-call"}, {"type": "token", "content": "parent final"}])
        starts = [event for event in events if event["type"] == "tool_start"]
        self.assertEqual([event["name"] for event in starts], ["subagent", "lookup"])
        self.assertEqual((starts[0]["label"], starts[0]["description"]), ("子代理", "Inspect sources"))
        self.assertEqual(starts[0]["tool_call_id"], "delegate-call")
        self.assertNotIn("parent_tool_call_id", starts[0])
        self.assertEqual(starts[1], {"type": "tool_start", "name": "lookup", "label": "lookup", "tool_call_id": "lookup-call", "parent_tool_call_id": "delegate-call"})
        ends = [event for event in events if event["type"] == "tool_end"]
        self.assertEqual([(event["name"], event["preview"]) for event in ends], [("lookup", "child lookup"), ("subagent", "child final")])
        self.assertEqual(ends[0]["parent_tool_call_id"], "delegate-call")
        self.assertEqual(ends[1]["tool_call_id"], "delegate-call")
        snapshot = await runner.get_state()
        self.assertEqual(len(snapshot.values["messages"]), 4)
        self.assertEqual(snapshot.values["messages"][-1].content, "parent final")
        self.assertEqual({entry.config["configurable"]["thread_id"] for entry in saver.list(None)}, {"parent"})
        self.assertNotIn("agent_scope", runner.config["metadata"])

    async def test_parallel_child_tokens_keep_their_parent_call_ids(self):
        child_model = RecordingModel(responses=[AIMessage(content="child final")])
        model = RecordingModel(responses=[
            AIMessage(content="", tool_calls=[
                {"name": "subagent", "args": {"description": "first", "prompt": "first task"}, "id": "first-call"},
                {"name": "subagent", "args": {"description": "second", "prompt": "second task"}, "id": "second-call"},
            ]),
            AIMessage(content="parent final"),
        ])
        registry = ToolRegistry()
        registry.register(build_subagent_tool(model=child_model), "子代理")
        runner = AgentRunner(
            context=BaseAgentContext(thread_id="parent", run_id="parent-run", user_id="owner"),
            model=model, tool_registry=registry, checkpointer=InMemorySaver(), store=None,
        )
        events = [event async for event in runner.run({"message": "Run both tasks"})]
        tokens = [event for event in events if event["type"] == "token"]
        self.assertEqual(len(tokens), 3)
        self.assertEqual({event["parent_tool_call_id"] for event in tokens[:2]}, {"first-call", "second-call"})
        self.assertTrue(all(event["content"] == "child final" for event in tokens[:2]))
        self.assertEqual(tokens[-1], {"type": "token", "content": "parent final"})
        self.assertNotIn("parent_tool_call_id", runner.config["metadata"])

    async def test_timeout_cancels_child_and_returns_a_tool_error(self):
        cancelled = asyncio.Event()

        async def hang(*args, **kwargs):
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        with patch(TARGET + ".AgentLoop", return_value=SimpleNamespace(run=hang)), patch(
            TARGET + ".settings.SUBAGENT_DISPATCH_TIMEOUT_SECONDS", 0.01,
        ):
            result = await invoke_tool(build_subagent_tool(model=Mock()), {"description": "Wait", "prompt": "Wait"})
        self.assertEqual(result.status, "error")
        self.assertIn("超时", result.content)
        self.assertTrue(cancelled.is_set())

    async def test_cancelling_parent_closes_child(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def hang(*args, **kwargs):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        with patch(TARGET + ".AgentLoop", return_value=SimpleNamespace(run=hang)):
            task = asyncio.create_task(invoke_tool(build_subagent_tool(model=Mock()), {"description": "Wait", "prompt": "Wait"}))
            try:
                await asyncio.wait_for(started.wait(), timeout=5)
            finally:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        self.assertTrue(cancelled.is_set())

    async def test_step_limit_and_empty_answers_return_tool_errors(self):
        @tool
        async def repeat() -> str:
            """Keep the child executing tool calls."""
            return "repeat"

        model = RecordingModel(responses=[AIMessage(content="", tool_calls=[{
            "name": "repeat", "args": {}, "id": "repeat-call",
        }])])
        with patch(TARGET + ".search_web_tool", repeat), patch(TARGET + ".SUBAGENT_RECURSION_LIMIT", 3):
            limited = await invoke_tool(build_subagent_tool(model=model), {"description": "Repeat", "prompt": "Repeat"})
        self.assertEqual(limited.status, "error")
        self.assertIn("步数上限", limited.content)
        empty = await invoke_tool(build_subagent_tool(model=RecordingModel(responses=[AIMessage(content=" ")])), {
            "description": "Empty", "prompt": "Empty",
        })
        self.assertEqual(empty.status, "error")
        self.assertIn("未生成最终回答", empty.content)


if __name__ == "__main__":
    unittest.main()
