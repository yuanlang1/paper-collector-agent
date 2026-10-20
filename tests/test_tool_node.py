import asyncio
import importlib
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langsmith import tracing_context

from app.llm.agent_build import build_tools
from app.llm.agent_runner import AgentRunContext
from app.llm.artifacts.access import ArtifactAccessService
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.loop.node.tool import build_tool_node
from app.llm.tools.artifact_tools.read_artifact import build_read_artifact_tool
from app.llm.tools.memory_tools.create_skill import CREATE_SKILL_TOOL
from app.llm.tools.memory_tools.update_soul import build_update_soul_tool
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
        context=context or AgentRunContext("thread-test", "run-test", "user-a"),
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
        )]}, context=AgentRunContext("thread", "run", "user"))
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
        tools = [
            SEARCH_WEB_TOOL, CREATE_SKILL_TOOL,
            build_read_artifact_tool(artifact_access=Mock()),
            build_update_soul_tool(session_factory=Mock()),
        ]
        for module_name, symbol, handler_name, args in cases:
            with self.subTest(tool=symbol):
                module = importlib.import_module("app.llm.tools." + module_name)
                selected = getattr(module, symbol)
                with patch.object(module, handler_name, new=AsyncMock(return_value={"ok": True})):
                    result = await invoke_tool(selected, args)
                self.assertEqual(result.status, "success")
                tools.append(selected)
        tools.append(build_search_paper_index_tool(content_retrieval=Mock()))
        tools.extend(build_tools(None, user_id=1).get_tools())
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
                invoke_tool(CREATE_SKILL_TOOL, args, context=AgentRunContext("thread-b", "run-b", "user-b")),
            )
            duplicate = await invoke_tool(CREATE_SKILL_TOOL, args)
        self.assertEqual(first.status, "success")
        self.assertEqual(second.status, "success")
        self.assertNotEqual(json.loads(first.content)["data"]["path"], json.loads(second.content)["data"]["path"])
        self.assertEqual(duplicate.status, "error")

    async def test_artifact_factories_keep_stores_separate_and_report_invalid_uri(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_store = LocalArtifactStore(first_dir)
            second_store = LocalArtifactStore(second_dir)
            first = await first_store.write_json(run_id="run-test", step_key="papers", source="test", kind="papers", payload={"papers": ["first"]})
            second = await second_store.write_json(run_id="run-test", step_key="papers", source="test", kind="papers", payload={"papers": ["second"]})
            first_tool = build_read_artifact_tool(artifact_access=ArtifactAccessService(first_store))
            second_tool = build_read_artifact_tool(artifact_access=ArtifactAccessService(second_store))
            results = await asyncio.gather(
                invoke_tool(first_tool, {"artifact_uri": first.artifact_uri, "mode": "json"}),
                invoke_tool(second_tool, {"artifact_uri": second.artifact_uri, "mode": "json"}),
            )
            unavailable = await invoke_tool(second_tool, {"artifact_uri": first.artifact_uri})
            invalid = await invoke_tool(first_tool, {"artifact_uri": "artifact://run/../outside.json"})
        self.assertTrue(all(result.status == "success" for result in results))
        self.assertEqual([json.loads(result.content)["data"]["data"]["value"] for result in results], [
            {"papers": ["first"]}, {"papers": ["second"]},
        ])
        self.assertEqual(unavailable.status, "error")
        self.assertEqual(invalid.status, "error")

    async def test_preferences_get_independent_sessions_and_runtime_user_identity(self):
        sessions = []
        def new_session():
            session = AsyncMock()
            session.__aenter__.return_value = session
            sessions.append(session)
            return session

        factory = Mock(side_effect=new_session)
        other_factory = Mock(side_effect=new_session)
        first_tool = build_update_soul_tool(session_factory=factory)
        second_tool = build_update_soul_tool(session_factory=other_factory)
        preference = SimpleNamespace(id="fact", content="Use Chinese")
        service = Mock(add=AsyncMock(return_value=(preference, True)))
        with patch("app.llm.tools.memory_tools.update_soul.UserPreferenceService", return_value=service) as constructor:
            results = await asyncio.gather(*(
                invoke_tool(selected, {"instruction": "Use Chinese"}, context=AgentRunContext(
                    "thread-" + user, "run-" + user, user,
                ))
                for selected, user in ((first_tool, "user-a"), (first_tool, "user-a"), (second_tool, "user-b"))
            ))
        self.assertTrue(all(result.status == "success" for result in results))
        self.assertEqual(factory.call_count, 2)
        other_factory.assert_called_once()
        self.assertEqual(len(sessions), 3)
        self.assertIsNot(sessions[0], sessions[1])
        self.assertIsNot(sessions[0], sessions[2])
        self.assertEqual({call.kwargs["user_id"] for call in constructor.call_args_list}, {"user-a", "user-b"})
        for session in sessions:
            session.__aexit__.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
