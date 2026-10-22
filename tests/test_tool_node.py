import asyncio
import importlib
import json
import tempfile
import unittest
from contextlib import aclosing
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from unittest.mock import AsyncMock, Mock, call, patch

import httpx
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langsmith import tracing_context

from app.llm.agent_build import build_tools
from app.llm.context import AgentRunContext
from app.llm.loop.node.tool import build_tool_node
from app.llm.tools.memory_tools.create_skill import CREATE_SKILL_TOOL
from app.llm.tools.search_tools.search_paper_index import build_search_paper_index_tool
from app.llm.tools.web_tools.search_web import SEARCH_WEB_TOOL


def tool_graph(tools):
    builder = StateGraph(MessagesState, context_schema=AgentRunContext)
    builder.add_node("tool", build_tool_node(tools))
    builder.add_edge(START, "tool")
    builder.add_edge("tool", END)
    return builder.compile()


async def invoke_tool(selected, args, *, context=None):
    output = await tool_graph([selected]).ainvoke(
        {"messages": [AIMessage(content="", tool_calls=[{
            "name": selected.name, "args": args, "id": "call-test",
        }])]},
        context=context or AgentRunContext(thread_id="thread-test", run_id="run-test", user_id="user-a"),
    )
    return output["messages"][-1]


class ToolNodeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(tracing_context(enabled=False))

    async def test_async_tools_execute_and_keep_each_call_id(self):
        @tool
        async def echo(value: str) -> dict:
            """Return a value without external IO."""
            return {"value": value}

        result = await tool_graph([echo]).ainvoke({"messages": [AIMessage(
            content="", tool_calls=[
                {"name": "echo", "args": {"value": "first"}, "id": "call-1"},
                {"name": "echo", "args": {"value": "second"}, "id": "call-2"},
            ],
        )]}, context=AgentRunContext(thread_id="thread", run_id="run", user_id="user"))
        messages = result["messages"][1:]
        self.assertTrue(all(isinstance(message, ToolMessage) for message in messages))
        self.assertEqual([message.status for message in messages], ["success", "success"])
        self.assertEqual([message.tool_call_id for message in messages], ["call-1", "call-2"])
        self.assertEqual([json.loads(message.content)["value"] for message in messages], ["first", "second"])

    async def test_expected_errors_keep_details_and_programming_errors_propagate(self):
        @tool
        async def expected_failure() -> str:
            """Report a recoverable tool failure."""
            raise ToolException("service unavailable")

        @tool
        async def programming_failure() -> str:
            """Raise an unexpected programming error."""
            raise TypeError("unexpected failure")

        result = await invoke_tool(expected_failure, {})
        self.assertEqual(result.status, "error")
        self.assertEqual(result.content, "service unavailable")
        with self.assertRaisesRegex(TypeError, "unexpected failure"):
            await invoke_tool(programming_failure, {})

    async def test_http_errors_do_not_expose_request_credentials(self):
        @tool
        async def http_failure() -> str:
            """Simulate an upstream HTTP error."""
            request = httpx.Request("GET", "https://example.test/?api_key=secret-value")
            response = httpx.Response(429, request=request)
            response.raise_for_status()

        result = await invoke_tool(http_failure, {})
        self.assertEqual(result.status, "error")
        self.assertIn("429", result.content)
        self.assertNotIn("secret-value", result.content)

    async def test_web_search_reaches_handler_and_validation_keeps_feedback(self):
        target = "app.llm.tools.web_tools.search_web"
        with patch(target + ".settings.FIRECRAWL_API_KEY", "test-key"), patch(
            target + "._post_search", new=AsyncMock(return_value={
                "success": True, "data": {"web": [{"title": "Result", "url": "https://example.test"}]},
            }),
        ) as search:
            result = await invoke_tool(SEARCH_WEB_TOOL, {"query": "RAG"})
            invalid = await invoke_tool(SEARCH_WEB_TOOL, {"query": "RAG", "limit": 99})
        self.assertEqual(result.status, "success")
        self.assertEqual(json.loads(result.content)["returned_count"], 1)
        search.assert_awaited_once()
        self.assertEqual(invalid.status, "error")
        self.assertIn("limit", invalid.content)
        with patch(target + ".settings.FIRECRAWL_API_KEY", ""):
            missing_key = await invoke_tool(SEARCH_WEB_TOOL, {"query": "RAG"})
        self.assertEqual(missing_key.status, "error")
        self.assertIn("API Key", missing_key.content)

    async def test_web_retries_are_silent_and_do_not_repeat_successful_parallel_tools(self):
        target = "app.llm.tools.web_tools.search_web"
        requests = []
        writes = []

        def respond(request):
            requests.append(request)
            if len(requests) < 3:
                return httpx.Response(503)
            return httpx.Response(200, json={
                "success": True, "data": {"web": [{"title": "Result", "url": "https://example.test"}]},
            })

        @tool
        async def write_once() -> str:
            """Record one successful write."""
            writes.append("saved")
            return "saved"

        client_type = httpx.AsyncClient
        transport = httpx.MockTransport(respond)
        sleep = asyncio.sleep

        async def yield_control(delay):
            await sleep(0)

        with patch(target + ".settings.FIRECRAWL_API_KEY", "test-secret"), patch(
            target + ".httpx.AsyncClient", side_effect=lambda **kwargs: client_type(transport=transport, **kwargs),
        ), patch("app.llm.loop.node.tool.asyncio.sleep", side_effect=yield_control) as waiting, patch(
            "app.llm.loop.node.tool.random.uniform", return_value=0,
        ):
            parts = [part async for part in tool_graph([SEARCH_WEB_TOOL, write_once]).astream(
                {"messages": [AIMessage(content="", tool_calls=[
                    {"name": "search_web", "args": {"query": "RAG"}, "id": "search-call"},
                    {"name": "write_once", "args": {}, "id": "write-call"},
                ])]}, stream_mode=["custom", "updates"], version="v2",
            )]
        self.assertEqual(len(requests), 3)
        self.assertEqual(writes, ["saved"])
        self.assertEqual(waiting.await_args_list, [call(1), call(2)])
        events = [part["data"] for part in parts if part["type"] == "custom"]
        self.assertEqual(len(events), 4)
        search_events = [event for event in events if event["name"] == "search_web"]
        self.assertEqual([event["type"] for event in search_events], ["tool_start", "tool_end"])
        self.assertEqual(search_events[0], {"type": "tool_start", "name": "search_web", "tool_call_id": "search-call"})
        self.assertEqual(search_events[-1]["status"], "success")
        self.assertEqual(set(search_events[-1]), {"type", "name", "status", "preview", "tool_call_id"})
        self.assertLess(events.index(next(event for event in events if event["type"] == "tool_end" and event["name"] == "write_once")), events.index(search_events[-1]))
        self.assertNotIn("test-secret", json.dumps(events))
        messages = next(part["data"]["tool"]["messages"] for part in parts if part["type"] == "updates")
        self.assertEqual([(message.tool_call_id, message.status) for message in messages], [("search-call", "success"), ("write-call", "success")])

    async def test_retries_stop_at_the_limit_and_permanent_errors_are_not_retried(self):
        request = httpx.Request("GET", "https://example.test/?key=secret")
        cases = [
            (httpx.HTTPStatusError("upstream failed", request=request, response=httpx.Response(code, request=request)), attempts)
            for code, attempts in ((429, 3), (500, 3), (502, 3), (503, 3), (504, 3), (401, 1), (402, 1), (403, 1), (404, 1))
        ] + [
            (httpx.ConnectError("offline", request=request), 3),
            (httpx.ReadTimeout("timed out", request=request), 3),
            (httpx.RemoteProtocolError("disconnected", request=request), 3),
            (httpx.UnsupportedProtocol("unsupported protocol", request=request), 1),
            (ToolException("configuration missing"), 1),
        ]
        for failure, attempts in cases:
            with self.subTest(error=type(failure).__name__, message=str(failure)):
                calls = []

                @tool("search_web")
                async def failing_search() -> str:
                    """Simulate an upstream search failure."""
                    calls.append("called")
                    raise failure

                with patch("app.llm.loop.node.tool.asyncio.sleep", new_callable=AsyncMock) as waiting:
                    parts = [part async for part in tool_graph([failing_search]).astream(
                        {"messages": [AIMessage(content="", tool_calls=[{"name": "search_web", "args": {}, "id": "search-call"}])]},
                        stream_mode="custom", version="v2",
                    )]
                events = [part["data"] for part in parts]
                self.assertEqual(len(calls), attempts)
                self.assertEqual(waiting.await_count, attempts - 1)
                self.assertEqual([event["type"] for event in events], ["tool_start", "tool_failed"])
                self.assertEqual(events[-1]["status"], "error")
                self.assertEqual(set(events[-1]), {"type", "name", "status", "preview", "tool_call_id"})
                self.assertEqual(events[-1]["tool_call_id"], "search-call")
                self.assertNotIn("key=secret", events[-1]["preview"])

    async def test_retry_after_controls_waiting_and_long_delays_stop_retries(self):
        now = datetime(2026, 10, 20, tzinfo=UTC)
        headers = (
            ("7", 7, 2),
            (format_datetime(now + timedelta(seconds=5), usegmt=True), 5, 2),
            ("invalid", 1, 2), ("inf", 1, 2), ("-1", 1, 2), ("60", None, 1),
        )
        for retry_after, delay, attempts in headers:
            with self.subTest(retry_after=retry_after):
                calls = []

                @tool("search_web")
                async def rate_limited_search() -> str:
                    """Respect the upstream retry delay."""
                    calls.append("called")
                    if len(calls) == 1:
                        request = httpx.Request("GET", "https://example.test")
                        httpx.Response(429, headers={"Retry-After": retry_after}, request=request).raise_for_status()
                    return "recovered"

                with patch("app.llm.loop.node.tool.datetime", wraps=datetime) as clock, patch(
                    "app.llm.loop.node.tool.random.uniform", return_value=0,
                ), patch("app.llm.loop.node.tool.asyncio.sleep", new_callable=AsyncMock) as waiting:
                    clock.now.return_value = now
                    result = await invoke_tool(rate_limited_search, {})
                self.assertEqual(len(calls), attempts)
                self.assertEqual(result.status, "success" if delay is not None else "error")
                if delay is None:
                    waiting.assert_not_awaited()
                else:
                    waiting.assert_awaited_once_with(delay)

    async def test_writes_error_messages_and_programming_errors_are_not_retried(self):
        calls = []

        @tool("add_query_task")
        async def write_task() -> str:
            """Simulate a write with an ambiguous upstream failure."""
            calls.append("write")
            request = httpx.Request("POST", "https://example.test")
            httpx.Response(503, request=request).raise_for_status()

        @tool("search_web")
        async def failed_result() -> ToolMessage:
            """Return a final business failure."""
            calls.append("error-result")
            return ToolMessage(content="no results", tool_call_id="call-test", status="error")

        @tool("search_web")
        async def programming_error() -> str:
            """Expose a programming error without repeating it."""
            calls.append("programming-error")
            raise TypeError("unexpected failure")

        with patch("app.llm.loop.node.tool.asyncio.sleep", new_callable=AsyncMock) as waiting:
            self.assertEqual((await invoke_tool(write_task, {})).status, "error")
            self.assertEqual((await invoke_tool(failed_result, {})).status, "error")
            with self.assertRaisesRegex(TypeError, "unexpected failure"):
                await invoke_tool(programming_error, {})
        self.assertEqual(calls, ["write", "error-result", "programming-error"])
        waiting.assert_not_awaited()

    async def test_cancelling_retry_wait_stops_without_more_calls_or_success_events(self):
        waiting = asyncio.Event()
        cancelled = asyncio.Event()
        calls = []
        events = []

        @tool("search_web")
        async def failing_search() -> str:
            """Fail before waiting for another attempt."""
            calls.append("called")
            raise httpx.ConnectError("offline")

        async def pause(delay):
            waiting.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        async def consume():
            async with aclosing(tool_graph([failing_search]).astream(
                {"messages": [AIMessage(content="", tool_calls=[{"name": "search_web", "args": {}, "id": "search-call"}])]},
                stream_mode="custom", version="v2",
            )) as stream:
                async for part in stream:
                    events.append(part["data"])

        with patch("app.llm.loop.node.tool.asyncio.sleep", side_effect=pause):
            task = asyncio.create_task(consume())
            try:
                await asyncio.wait_for(waiting.wait(), timeout=5)
            finally:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        self.assertTrue(cancelled.is_set())
        self.assertEqual(calls, ["called"])
        self.assertEqual([event["type"] for event in events], ["tool_start"])

    async def test_ordinary_tools_load_and_run_with_native_schemas(self):
        cases = (
            ("search_tools.arxiv.search_arxiv", "ARXIV_SEARCH_TOOL", "arxiv_search_service", {"query": "RAG"}),
            ("search_tools.dblp.search_dblp", "DBLP_SEARCH_TOOL", "dblp_search_service", {"query": "RAG"}),
            ("search_tools.crossref.search_crossref", "CROSSREF_SEARCH_TOOL", "crossref_search_service", {"doi": "10.1234/test"}),
            ("search_tools.google_scholar.search_google_scholar", "GOOGLE_SCHOLAR_SEARCH_TOOL", "google_scholar_search_service", {"query": "RAG"}),
            ("search_tools.arxiv.supplement", "ARXIV_SUPPLEMENT_TOOL", "arxiv_supplement_handler", {"papers": []}),
            ("task_tools.search_task.search_task_create", "ADD_QUERY_TASK_TOOL", "add_query_task_handler", {
                "prompt": "Find RAG papers", "searchTag": {},
                "queryUnderstanding": {"topic": "RAG", "intent": "survey", "reasoning": "User requested a search"},
            }),
            ("task_tools.search_task.get_task_papers", "GET_TASK_PAPERS_TOOL", "get_task_papers_handler", {"task_id": 1}),
            ("task_tools.search_task.rag_status", "GET_TASK_RAG_STATUS_TOOL", "get_task_rag_status_handler", {"task_id": 1}),
            ("venue_tools.easy_scholar", "EASY_SCHOLAR_VENUE_TOOL", "easy_scholar_venue_handler", {"publication_name": "Nature"}),
            ("file_tools.download_file", "DOWNLOAD_FILE_TOOL", "download_file_handler", {
                "url": "https://example.test/paper.pdf", "save_dir": "papers", "file_name": "paper.pdf",
            }),
        )
        tools = [SEARCH_WEB_TOOL, CREATE_SKILL_TOOL]
        for module_name, symbol, handler_name, args in cases:
            with self.subTest(tool=symbol):
                module = importlib.import_module("app.llm.tools." + module_name)
                selected = getattr(module, symbol)
                with patch.object(module, handler_name, new=AsyncMock(return_value={"ok": True})):
                    result = await invoke_tool(selected, args)
                self.assertEqual(result.status, "success")
                tools.append(selected)
        tools.append(build_search_paper_index_tool(content_retrieval=Mock()))
        tools.extend(build_tools(None, user_id=1, model=Mock()).get_tools())
        for selected in tools:
            self.assertNotIn("requires_confirmation", selected.metadata or {})
            properties = convert_to_openai_tool(selected)["function"]["parameters"]["properties"]
            self.assertTrue({"runtime", "config", "db", "user_id", "session_factory", "artifact_access"}.isdisjoint(properties))

    async def test_skill_runtime_keeps_users_separate_and_duplicate_is_a_tool_error(self):
        args = {"name": "review", "description": "Review papers", "triggers": ["review"], "instructions": "Check evidence."}
        with tempfile.TemporaryDirectory() as directory, patch(
            "app.llm.tools.memory_tools.create_skill.settings.AGENT_SKILLS_DIR", directory,
        ):
            first, second = await asyncio.gather(
                invoke_tool(CREATE_SKILL_TOOL, args),
                invoke_tool(CREATE_SKILL_TOOL, args, context=AgentRunContext(thread_id="thread-b", run_id="run-b", user_id="user-b")),
            )
            duplicate = await invoke_tool(CREATE_SKILL_TOOL, args)
        self.assertEqual(first.status, "success")
        self.assertEqual(second.status, "success")
        self.assertNotEqual(json.loads(first.content)["data"]["path"], json.loads(second.content)["data"]["path"])
        self.assertEqual(duplicate.status, "error")


if __name__ == "__main__":
    unittest.main()
