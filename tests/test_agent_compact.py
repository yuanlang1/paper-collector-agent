import json
import unittest
from unittest.mock import Mock

import httpx
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langsmith import tracing_context
from pydantic import Field

from app.llm.agent_runner import AgentRunner
from app.llm.context import AgentRunContext, BaseAgentContext
from app.llm.loop.agent_loop import AgentLoop
from app.llm.loop.node.compact import (
    CompactNode, active_messages, compaction_messages, model_messages,
)
from app.llm.tools.registry import ToolRegistry


class RecordingModel(FakeMessagesListChatModel):
    streaming: bool = True
    calls: list[dict] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append({"stream": False, "messages": list(messages), **kwargs})
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append({"stream": True, "messages": list(messages), **kwargs})
        response = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs).generations[0].message
        yield ChatGenerationChunk(message=AIMessageChunk(
            content=response.content,
            tool_call_chunks=[{
                "name": call["name"], "args": json.dumps(call["args"]),
                "id": call["id"], "index": index,
            } for index, call in enumerate(response.tool_calls)],
            usage_metadata=response.usage_metadata,
        ))


@tool
async def lookup(value: str) -> str:
    """Look up a value for the current task."""
    return "result:" + value


def tool_call(value: str, call_id: str = "call-1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{
        "name": "lookup", "args": {"value": value}, "id": call_id,
    }], usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})


def conversation():
    return {"messages": [
        HumanMessage(content="old task", id="old-user"),
        AIMessage(content="old answer", id="old-answer"),
        HumanMessage(content="current task", id="current-user"),
        AIMessage(content="", id="current-call", tool_calls=[{
            "name": "lookup", "args": {"value": "current"}, "id": "call-current",
        }]),
        ToolMessage(content="new tool output", tool_call_id="call-current", id="current-result"),
    ]}


class CompactTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(tracing_context(enabled=False))

    def context(self, **kwargs):
        return AgentRunContext(**{
            "thread_id": "thread", "run_id": "run", "user_id": "1",
            "keep_last_n_turns": 1, "context_window_tokens": 1_000,
            **kwargs,
        })

    async def test_new_tool_output_triggers_compaction_despite_low_previous_usage(self):
        @tool
        async def large_lookup() -> str:
            """Return a large result from a short tool request."""
            return "result:" + "x" * 5_000

        request = AIMessage(content="", tool_calls=[{
            "name": "large_lookup", "args": {}, "id": "large-call",
        }], usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
        model = RecordingModel(streaming=True, responses=[
            request, AIMessage(content="cumulative summary"), AIMessage(content="done"),
        ])
        loop = AgentLoop(
            context=self.context(auto_threshold=0.8), model=model, tools=[large_lookup],
            system_prompt="agent instructions",
        )
        output = await loop.run([
            HumanMessage(content="old task"), AIMessage(content="old answer"),
            HumanMessage(content="look up a large result"),
        ])
        self.assertEqual(len(output["summaries"]), 1)
        self.assertEqual(output["summaries"][0]["end_id"], output["messages"][1].id)
        self.assertLess(count_tokens_approximately(
            model.calls[0]["messages"] + [request], tools=[large_lookup],
        ), 800)
        self.assertEqual(output["messages"][-2].content, "result:" + "x" * 5_000)
        self.assertEqual(output["messages"][-1].content, "done")

    async def test_threshold_is_strict_and_disabled_or_retained_only_history_is_skipped(self):
        state = conversation()
        model = Mock(profile=None)
        node = CompactNode(model=model, tools=[lookup], system_prompt="instructions")
        tokens = count_tokens_approximately(model_messages(state, "instructions"), tools=[lookup])
        context = self.context(auto_threshold=0.5, context_window_tokens=tokens * 2)
        self.assertEqual(node.route(state, Runtime(context=context)), "solve")
        state["messages"][-1].content += "x" * 100
        self.assertEqual(node.route(state, Runtime(context=context)), "compact")
        self.assertEqual(node.route(state, Runtime(context=self.context(
            auto_threshold=0, context_window_tokens=None,
        ))), "solve")
        self.assertEqual(node.route(state, Runtime(context=self.context(
            keep_last_n_turns=2, context_window_tokens=None,
        ))), "solve")
        model.ainvoke.assert_not_called()

    async def test_counter_includes_prompt_summary_tools_and_returned_results(self):
        state = conversation()
        state["summaries"] = [{
            "content": "earlier summary", "start_id": "old-user", "end_id": "old-answer", "created_at": "now",
        }]
        # Keep two active turns so there is an older turn available to compact.
        state["messages"].extend([
            AIMessage(content="finished", id="current-answer"),
            HumanMessage(content="next task", id="next-user"),
            tool_call("next", "call-next"),
            ToolMessage(content="latest output", tool_call_id="call-next", id="next-result"),
        ])
        baseline = count_tokens_approximately(model_messages(state, "instructions"), tools=[lookup])
        node = CompactNode(model=Mock(profile=None), tools=[lookup], system_prompt="instructions")
        context = self.context(auto_threshold=0.5, context_window_tokens=baseline * 2 - 1)
        self.assertEqual(node.route(state, Runtime(context=context)), "compact")
        for changed in (
            CompactNode(model=Mock(profile=None), tools=[lookup]),
            CompactNode(model=Mock(profile=None), system_prompt="instructions"),
        ):
            self.assertEqual(changed.route(state, Runtime(context=context)), "solve")
        self.assertEqual(model_messages(state, "instructions")[-1].content, "latest output")
        self.assertEqual(model_messages(state, "instructions")[1].content, "earlier summary")

    async def test_retention_uses_whole_user_turns(self):
        state = conversation()
        state["messages"].extend([
            AIMessage(content="answer", id="current-answer"),
            HumanMessage(content="third task", id="third-user"),
            AIMessage(content="", id="third-calls", tool_calls=[
                {"name": "lookup", "args": {"value": "a"}, "id": "a"},
                {"name": "lookup", "args": {"value": "b"}, "id": "b"},
            ]),
            ToolMessage(content="a", tool_call_id="a", id="a-result"),
            ToolMessage(content="b", tool_call_id="b", id="b-result"),
        ])
        self.assertEqual([m.id for m in compaction_messages(state, 2)], ["old-user", "old-answer"])
        self.assertEqual(compaction_messages(state, 3), [])
        self.assertEqual(compaction_messages(state, 1)[-1].id, "current-answer")

    async def test_cumulative_summaries_survive_checkpoint_and_do_not_duplicate_history(self):
        model = RecordingModel(streaming=True, responses=[
            AIMessage(content="answer one"),
            tool_call("two", "call-two"), AIMessage(content="summary one"), AIMessage(content="answer two"),
            tool_call("three", "call-three"), AIMessage(content="summary two"), AIMessage(content="answer three"),
        ])
        loop = AgentLoop(context=self.context(context_window_tokens=1), model=model, tools=[lookup], checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "thread"}}
        await loop.run([("human", "task one")], config=config)
        second = await loop.run([("human", "task two")], config=config)
        third = await loop.run([("human", "task three")], config=config)
        self.assertEqual(len(third["messages"]), 10)
        self.assertEqual(len(third["summaries"]), 2)
        self.assertEqual(third["summaries"][1]["start_id"], third["messages"][0].id)
        self.assertEqual(third["summaries"][1]["end_id"], third["messages"][5].id)
        self.assertEqual(second["summaries"][0]["end_id"], third["messages"][1].id)
        self.assertEqual(active_messages(third)[0].content, "task three")
        inputs = model.calls[-1]["messages"]
        self.assertEqual(inputs[0].content, "summary two")
        self.assertEqual(inputs[1].content, "task three")
        self.assertEqual([m.content for m in inputs if isinstance(m, ToolMessage)], ["result:three"])
        compact_calls = [call for call in model.calls if not call["stream"]]
        self.assertEqual(len(compact_calls), 2)
        self.assertTrue(all("tools" not in call for call in compact_calls))
        history = json.loads(compact_calls[-1]["messages"][-1].content)
        self.assertEqual(history["previous_summary"], "summary one")
        self.assertEqual(history["history"][0]["content"], "task two")

    async def test_compact_payload_has_no_tools_and_is_non_streaming(self):
        requests = []

        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "compact", "object": "chat.completion", "created": 0, "model": "test-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "summary"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            model = ChatOpenAI(
                model="test-model", api_key="test-key", base_url="https://example.invalid/v1",
                streaming=True, http_async_client=client,
            )
            node = CompactNode(model=model, tools=[lookup])
            state = conversation()
            output = await node(state, {}, Runtime(context=self.context()))
        self.assertIs(node.model, model)
        self.assertTrue(model.streaming)
        self.assertFalse(requests[0]["stream"])
        self.assertNotIn("tools", requests[0])
        self.assertTrue(all("tool_calls" not in message for message in requests[0]["messages"]))
        self.assertEqual(output["summaries"][0]["content"], "summary")
        self.assertEqual(len(state["messages"]), 5)

    async def test_runner_streams_solve_but_does_not_publish_compact_text(self):
        model = RecordingModel(streaming=True, responses=[
            AIMessage(content="first answer"), tool_call("value"),
            AIMessage(content="PRIVATE SUMMARY"), AIMessage(content="final answer"),
        ])
        registry = ToolRegistry()
        registry.register(lookup, "查询")
        saver = InMemorySaver()
        first = AgentRunner(context=self.context(context_window_tokens=1), model=model, tool_registry=registry, checkpointer=saver, store=None)
        _ = [event async for event in first.run({"message": "old task"})]
        runner = AgentRunner(context=self.context(run_id="next", context_window_tokens=1), model=model, tool_registry=registry, checkpointer=saver, store=None)
        events = [event async for event in runner.run({"message": "new task"})]
        self.assertEqual("".join(event["content"] for event in events if event["type"] == "token"), "final answer")
        self.assertNotIn("PRIVATE SUMMARY", json.dumps(events))
        self.assertTrue(model.streaming)
        self.assertTrue(model.calls[-1]["stream"])
        self.assertIn("tools", model.calls[-1])

    async def test_missing_window_skips_compaction_and_profile_window_still_applies(self):
        context = self.context(context_window_tokens=None)
        for profile in (None, {}, {"max_input_tokens": None}, {"max_input_tokens": 0}):
            with self.subTest(profile=profile):
                model = Mock(profile=profile)
                node = CompactNode(model=model)
                self.assertEqual(node.route(conversation(), Runtime(context=context)), "solve")
                model.ainvoke.assert_not_called()
        node = CompactNode(model=Mock(profile={"max_input_tokens": 1}))
        self.assertEqual(node.route(conversation(), Runtime(context=context)), "compact")

    async def test_missing_window_keeps_tool_replies_in_checkpoint_across_turns(self):
        model = RecordingModel(responses=[
            AIMessage(content="first answer"), AIMessage(content="second answer"),
            AIMessage(content="", tool_calls=[
                {"name": "lookup", "args": {"value": "a"}, "id": "call-a"},
                {"name": "lookup", "args": {"value": "b"}, "id": "call-b"},
            ]),
            AIMessage(content="third answer"), AIMessage(content="fourth answer"),
        ])
        registry = ToolRegistry()
        registry.register(lookup, "查询")
        runner = AgentRunner(
            context=self.context(context_window_tokens=None, keep_last_n_turns=2),
            model=model, tool_registry=registry, checkpointer=InMemorySaver(), store=None,
        )
        for turn in ("first", "second", "third", "fourth"):
            events = [event async for event in runner.run({"message": turn})]
            self.assertEqual([event["content"] for event in events if event["type"] == "token"], [turn + " answer"])
        snapshot = await runner.get_state()
        self.assertFalse(snapshot.next)
        self.assertFalse(snapshot.values.get("summaries"))
        self.assertEqual([(message.tool_call_id, message.content) for message in snapshot.values["messages"] if isinstance(message, ToolMessage)], [
            ("call-a", "result:a"), ("call-b", "result:b"),
        ])
        messages = model.calls[-1]["messages"]
        call_index = next(index for index, message in enumerate(messages) if isinstance(message, AIMessage) and message.tool_calls)
        self.assertEqual([message.tool_call_id for message in messages[call_index + 1:call_index + 3]], ["call-a", "call-b"])

    async def test_invalid_configuration_and_empty_summary_fail_explicitly(self):
        for fields in (
            {"auto_threshold": -0.1}, {"auto_threshold": 1.1},
            {"keep_last_n_turns": 0}, {"keep_last_n_turns": 1.5}, {"context_window_tokens": 0},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.context(**fields)
        state = conversation()
        node = CompactNode(model=RecordingModel(responses=[AIMessage(content=" ")]))
        with self.assertRaisesRegex(ValueError, "有效摘要"):
            await node(state, {}, Runtime(context=self.context()))
        self.assertNotIn("summaries", state)
