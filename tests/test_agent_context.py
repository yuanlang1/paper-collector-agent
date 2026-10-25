import json
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain.tools import ToolRuntime
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
from langsmith import tracing_context

from app.bus.events import SubagentProgressEvent
from app.llm.agent_runner import AgentRunner
from app.llm.context import AgentRunContext, BaseAgentContext
from app.llm.loop.agent_loop import AgentLoop
from app.llm.tools.registry import ToolRegistry
from app.services import agent_run_service


@dataclass(kw_only=True)
class TaskContext(BaseAgentContext):
    task_id: int


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class AgentContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(tracing_context(enabled=False))

    async def test_loop_run_and_stream_inject_the_supplied_context_type(self):
        observed = []

        @tool
        async def read_context(runtime: ToolRuntime[BaseAgentContext]) -> str:
            """Read the current agent context."""
            observed.append(runtime.context)
            return runtime.context.thread_id

        contexts = (
            AgentRunContext(thread_id="main", run_id="run-main", user_id="1"),
            TaskContext(thread_id="task", run_id="run-task", user_id="2", task_id=7),
        )
        for context in contexts:
            for method in ("run", "stream"):
                with self.subTest(context=type(context).__name__, method=method):
                    model = ToolCallingModel(responses=[
                        AIMessage(content="", tool_calls=[{
                            "name": "read_context", "args": {}, "id": "call-1",
                        }]),
                        AIMessage(content="done"),
                    ])
                    loop = AgentLoop(context=context, model=model, tools=[read_context])
                    if method == "run":
                        output = await loop.run([("human", "Read the context")])
                        self.assertEqual(output["messages"][-2].content, context.thread_id)
                    else:
                        events = [event async for event in loop.stream([("human", "Read the context")])]
                        result = next(part["data"] for part in events if part["type"] == "custom" and part["data"]["type"] == "tool_end")
                        self.assertEqual(result["preview"], context.thread_id)
                    self.assertIs(observed[-1], context)
                    self.assertIs(loop.graph.context_schema, type(context))

    async def test_runner_passes_custom_context_and_matching_config_to_tools(self):
        context = TaskContext(
            thread_id="thread-task", run_id="run-task", user_id="2",
            task_id=7, auto_threshold=0.6,
        )
        observed = []

        @tool
        async def read_task(runtime: ToolRuntime[TaskContext]) -> str:
            """Read the task supplied by the agent caller."""
            observed.append(runtime)
            return str(runtime.context.task_id)

        registry = ToolRegistry()
        registry.register(read_task, "读取任务")
        runner = AgentRunner(
            context=context,
            model=ToolCallingModel(responses=[
                AIMessage(content="", tool_calls=[{
                    "name": "read_task", "args": {}, "id": "call-1",
                }]),
                AIMessage(content="done"),
            ]),
            tool_registry=registry, checkpointer=InMemorySaver(), store=None,
        )
        events = [event async for event in runner.run({"message": "Read the task"})]
        self.assertIs(observed[0].context, context)
        self.assertEqual(observed[0].config["configurable"]["thread_id"], context.thread_id)
        self.assertEqual(observed[0].config["metadata"]["user_id"], context.user_id)
        self.assertEqual(observed[0].config["metadata"]["app_run_id"], context.run_id)
        result = next(event for event in events if event["type"] == "tool_end")
        self.assertEqual((result["label"], result["preview"]), ("读取任务", "7"))

    async def test_service_creates_and_injects_the_main_agent_context(self):
        observed = []

        @tool
        async def read_identity(runtime: ToolRuntime[AgentRunContext]) -> str:
            """Read the identity supplied by the run service."""
            observed.append(runtime.context)
            return runtime.context.user_id

        registry = ToolRegistry()
        registry.register(read_identity, "读取身份")
        model = ToolCallingModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "read_identity", "args": {}, "id": "call-1",
            }]),
            AIMessage(content="done"),
        ])
        run = SimpleNamespace(
            thread_id="thread", run_id="run", user_id=1, run_metadata={"model_id": 7},
        )
        with patch.object(agent_run_service, "update_status", new=AsyncMock(return_value=run)), patch.object(
            agent_run_service, "complete_run", new=AsyncMock(return_value=(run, [])),
        ), patch.object(
            agent_run_service.agent_build, "get_model", new=AsyncMock(return_value=model),
        ), patch.object(agent_run_service.agent_build, "build_tools", return_value=registry):
            events = [event async for event in agent_run_service.run(
                db=None, run=run, input={"message": "Read the identity"},
                checkpointer=InMemorySaver(), store=None,
            )]
        self.assertIs(type(observed[0]), AgentRunContext)
        self.assertEqual(observed[0], AgentRunContext(thread_id="thread", run_id="run", user_id="1"))
        self.assertEqual(events[0], {"type": "status", "status": "running"})
        self.assertEqual(events[-1], {"type": "done", "status": "completed", "interrupts": []})
        for event in events:
            self.assertTrue(set(event) <= {"type", "status", "message", "content", "name", "label", "preview", "description", "interrupts", "tool_call_id", "parent_tool_call_id"})

    async def test_runner_forwards_custom_progress_as_json_without_repeating_tool_text(self):
        @tool
        async def report_progress(runtime: ToolRuntime) -> str:
            """Emit structured progress while executing a tool."""
            runtime.stream_writer(SubagentProgressEvent(
                session_id="session", run_id="child", ts="2026-10-20T00:00:00Z",
                parent_run_id="run", parent_session_id="session", name="worker",
                description="Processing sources",
            ))
            runtime.stream_writer({
                "type": "subagent_progress", "description": "Saving results", "task_id": 7,
                "thread_id": "thread", "run_id": "run", "attempt": 2, "attempts": 3,
                "parent_tool_call_id": "progress-call", "tool_call_id": "inner-call",
                "namespace": ["tool:task"], "agent_scope": "subagent",
            })
            return "tool result"

        registry = ToolRegistry()
        registry.register(report_progress, "报告进度")
        runner = AgentRunner(
            context=BaseAgentContext(thread_id="thread", run_id="run", user_id="1"),
            model=ToolCallingModel(responses=[
                AIMessage(content="", tool_calls=[{"name": "report_progress", "args": {}, "id": "progress-call"}]),
                AIMessage(content="model final"),
            ]),
            tool_registry=registry, checkpointer=InMemorySaver(), store=None,
        )
        events = [event async for event in runner.run({"message": "Report progress"})]
        progress = [event for event in events if event["type"] == "subagent_progress"]
        self.assertEqual(progress, [
            {"type": "subagent_progress", "name": "worker", "description": "Processing sources", "parent_tool_call_id": "progress-call"},
            {"type": "subagent_progress", "description": "Saving results", "task_id": 7, "parent_tool_call_id": "progress-call", "tool_call_id": "inner-call"},
        ])
        self.assertEqual([event["content"] for event in events if event["type"] == "token"], ["model final"])
        json.dumps(events)

    async def test_service_terminal_events_omit_run_and_thread_ids(self):
        async def stream(input):
            yield {"type": "token", "content": "waiting"}

        run = SimpleNamespace(thread_id="thread", run_id="run", user_id=1, run_metadata={"model_id": 7})
        snapshot = SimpleNamespace(
            tasks=(), next=("tool",), interrupts=(SimpleNamespace(id="approval", value={"question": "approve?"}),),
        )
        runner = SimpleNamespace(run=stream, get_state=AsyncMock(return_value=snapshot))
        for failed in (False, True):
            with self.subTest(failed=failed), patch.object(
                agent_run_service, "update_status", new=AsyncMock(return_value=run),
            ), patch.object(
                agent_run_service.agent_build, "get_model", new=AsyncMock(return_value=object()),
            ), patch.object(agent_run_service.agent_build, "build_tools"), patch.object(
                agent_run_service, "AgentRunner", return_value=runner,
            ), patch.object(agent_run_service.logger, "exception"):
                runner.get_state.side_effect = RuntimeError("snapshot failed") if failed else None
                events = [event async for event in agent_run_service.run(
                    db=None, run=run, input={"message": "Wait for approval"}, checkpointer=None, store=None,
                )]
            for event in events:
                self.assertNotIn("thread_id", event)
                self.assertNotIn("run_id", event)
            if failed:
                self.assertEqual(events[-1], {"type": "error", "status": "failed", "message": "大模型调用失败，请稍后重试"})
            else:
                self.assertEqual(events[-1], {"type": "done", "status": "interrupted", "interrupts": [{"id": "approval", "value": {"question": "approve?"}}]})
                self.assertEqual(events[-2], {**events[-1], "type": "interrupt"})

    async def test_tool_subgraph_inherits_call_id_without_special_tool_logic(self):
        @tool
        async def child_work(runtime: ToolRuntime) -> str:
            """Run a graph from inside an ordinary tool."""
            child = AgentLoop(
                context=runtime.context, model=ToolCallingModel(responses=[AIMessage(content="child text")]),
                checkpointer=False,
            )
            output = await child.run([("human", "Child task")], config=runtime.config)
            return output["messages"][-1].text

        registry = ToolRegistry()
        registry.register(child_work, "子图任务")
        runner = AgentRunner(
            context=BaseAgentContext(thread_id="thread", run_id="run", user_id="1"),
            model=ToolCallingModel(responses=[
                AIMessage(content="", tool_calls=[{"name": "child_work", "args": {}, "id": "graph-call"}]),
                AIMessage(content="parent text"),
            ]), tool_registry=registry, checkpointer=InMemorySaver(), store=None,
        )
        events = [event async for event in runner.run({"message": "Run child graph"})]
        self.assertEqual([event for event in events if event["type"] == "token"], [
            {"type": "token", "content": "child text", "parent_tool_call_id": "graph-call"},
            {"type": "token", "content": "parent text"},
        ])
        self.assertEqual([event["tool_call_id"] for event in events if event["type"] in ("tool_start", "tool_end")], ["graph-call", "graph-call"])
        self.assertNotIn("parent_tool_call_id", runner.config["metadata"])

    async def test_final_tool_failures_emit_failed_events_and_keep_error_behavior(self):
        for failure in ("expected", "returned", "validation", "unexpected"):
            with self.subTest(failure=failure):
                @tool
                async def failing_tool(value: str, runtime: ToolRuntime) -> str | ToolMessage:
                    """Report a final tool execution failure."""
                    if failure == "expected":
                        raise ToolException("service unavailable")
                    if failure == "unexpected":
                        raise TypeError("private programming details")
                    return ToolMessage(content="business failure", tool_call_id=runtime.tool_call_id, status="error")

                registry = ToolRegistry()
                registry.register(failing_tool, "失败工具")
                runner = AgentRunner(
                    context=BaseAgentContext(thread_id="thread", run_id="run", user_id="1"),
                    model=ToolCallingModel(responses=[
                        AIMessage(content="", tool_calls=[{
                            "name": "failing_tool", "args": {} if failure == "validation" else {"value": "input"}, "id": "failed-call",
                        }]), AIMessage(content="done"),
                    ]), tool_registry=registry, checkpointer=InMemorySaver(), store=None,
                )
                events = []

                async def collect():
                    async for event in runner.run({"message": "Run the tool"}):
                        events.append(event)

                if failure == "unexpected":
                    with self.assertRaisesRegex(TypeError, "private programming details"):
                        await collect()
                else:
                    await collect()
                    snapshot = await runner.get_state()
                    result = next(message for message in snapshot.values["messages"] if isinstance(message, ToolMessage))
                    self.assertEqual((result.tool_call_id, result.status), ("failed-call", "error"))
                tool_events = [event for event in events if event["type"].startswith("tool_")]
                self.assertEqual([event["type"] for event in tool_events], ["tool_start", "tool_failed"])
                self.assertTrue(all(event["tool_call_id"] == "failed-call" and event["label"] == "失败工具" for event in tool_events))
                self.assertEqual(tool_events[-1]["status"], "error")
                self.assertNotIn("private programming details", tool_events[-1]["preview"])

    async def test_resume_uses_the_new_run_context(self):
        observed = []

        @tool
        async def confirm_task(runtime: ToolRuntime[TaskContext]) -> str:
            """Wait for a decision about the current task."""
            decision = interrupt({"task_id": runtime.context.task_id})
            observed.append(runtime.context)
            return decision

        registry = ToolRegistry()
        registry.register(confirm_task, "确认任务")
        saver = InMemorySaver()
        context = TaskContext(thread_id="thread", run_id="first", user_id="1", task_id=7)
        runner = AgentRunner(
            context=context,
            model=ToolCallingModel(responses=[AIMessage(content="", tool_calls=[{
                "name": "confirm_task", "args": {}, "id": "call-1",
            }])]),
            tool_registry=registry, checkpointer=saver, store=None,
        )
        paused_events = [event async for event in runner.run({"message": "Confirm the task"})]
        self.assertEqual(paused_events[0], {"type": "status", "message": "正在思考…"})
        self.assertNotIn("tool_failed", [event["type"] for event in paused_events])
        self.assertTrue((await runner.get_state()).interrupts)

        resumed_context = TaskContext(thread_id="thread", run_id="second", user_id="1", task_id=7)
        resumed = AgentRunner(
            context=resumed_context,
            model=ToolCallingModel(responses=[AIMessage(content="done")]),
            tool_registry=registry, checkpointer=saver, store=None,
        )
        events = [event async for event in resumed.run({"resume": "approved"})]
        self.assertFalse(any(event["type"] == "status" for event in events))
        self.assertIs(observed[0], resumed_context)
        self.assertEqual(next(event for event in events if event["type"] == "tool_end")["preview"], "approved")
        self.assertFalse((await resumed.get_state()).interrupts)
        self.assertEqual((await resumed.get_state()).values["messages"][-1].content, "done")

    async def test_hitl_tools_resume_approve_edit_and_reject(self):
        for decision, expected_city, expected_result in (
            ({"type": "approve"}, "北京", "北京天气晴朗"),
            ({"type": "edit", "edited_action": {"name": "get_weather", "args": {"city": "上海"}}}, "上海", "上海天气晴朗"),
            ({"type": "reject", "message": "取消查询"}, None, "取消查询"),
        ):
            with self.subTest(decision=decision):
                executed = []
                hitl_request = {
                    "action_requests": [{
                        "name": "get_weather", "args": {"city": "北京"}, "description": "请审核查询城市",
                    }],
                    "review_configs": [{
                        "action_name": "get_weather", "allowed_decisions": ["approve", "edit", "reject"],
                    }],
                }

                @tool
                async def get_weather(city: str) -> str:
                    """Review the city before querying the weather."""
                    response = interrupt(hitl_request)
                    answer = response["decisions"][0]
                    if answer["type"] == "reject":
                        return answer["message"]
                    if answer["type"] == "edit":
                        city = answer["edited_action"]["args"]["city"]
                    executed.append(city)
                    return f"{city}天气晴朗"

                registry = ToolRegistry()
                registry.register(get_weather, "查询天气")
                model = ToolCallingModel(responses=[
                    AIMessage(content="", tool_calls=[{
                        "name": "get_weather", "args": {"city": "北京"}, "id": "weather-call",
                    }]),
                    AIMessage(content="done"),
                ])
                saver = InMemorySaver()
                run = SimpleNamespace(thread_id="thread", run_id="first", user_id=1, run_metadata={"model_id": 7})
                with patch.object(
                    agent_run_service, "update_status", new=AsyncMock(return_value=run),
                ), patch.object(
                    agent_run_service, "complete_run", new=AsyncMock(return_value=(run, [])),
                ), patch.object(
                    agent_run_service.agent_build, "get_model", new=AsyncMock(return_value=model),
                ), patch.object(agent_run_service.agent_build, "build_tools", return_value=registry):
                    paused = [event async for event in agent_run_service.run(
                        db=None, run=run, input={"message": "查询天气"}, checkpointer=saver, store=None,
                    )]
                    self.assertEqual(paused[:2], [
                        {"type": "status", "status": "running"},
                        {"type": "status", "message": "正在思考…"},
                    ])
                    self.assertEqual(executed, [])
                    self.assertEqual(paused[-1]["status"], "interrupted")
                    self.assertEqual(paused[-1]["interrupts"][0]["value"], hitl_request)
                    self.assertNotIn("tool_failed", [event["type"] for event in paused])
                    resumed_run = SimpleNamespace(**{**vars(run), "run_id": "second"})
                    resumed = [event async for event in agent_run_service.run(
                        db=None, run=resumed_run, input={"resume": {"decisions": [decision]}},
                        checkpointer=saver, store=None,
                    )]
                self.assertEqual(executed, [expected_city] if expected_city else [])
                self.assertFalse(any(event["type"] == "status" for event in resumed))
                result = next(event for event in resumed if event["type"] == "tool_end")
                self.assertEqual((result["tool_call_id"], result["preview"]), ("weather-call", expected_result))
                self.assertEqual(resumed[-1], {"type": "done", "status": "completed", "interrupts": []})
