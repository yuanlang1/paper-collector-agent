import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langsmith import tracing_context

from app.llm.agent_build import build_tools
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.context import BaseAgentContext
from app.llm.graph.workflows.paper_search.nodes.abstract_enrich import AbstractEnrichNode
from app.llm.graph.workflows.paper_search.nodes.plan_search import PlanSearchNode
from app.llm.graph.workflows.paper_search.nodes.recommend import RecommendationNode
from app.llm.graph.workflows.paper_search.nodes.review import ReviewNode
from app.llm.graph.workflows.paper_search.nodes.venue import VenueCacheStore, VenueResolutionNode
from app.llm.graph.workflows.paper_search.workflow import build_paper_search_workflow
from app.llm.graph.workflows.paper_search_schemas import PromptUnderstandingArgs
from app.llm.provider import StructuredOutputError, ValidatedJsonInvoker


UNDERSTANDING = {"topic": "retrieval augmented generation", "intent": "method", "reasoning": "检索方法论文。"}
PLAN = {
    "query_understanding": UNDERSTANDING,
    "query_plan": {
        "display_query": "retrieval augmented generation",
        "reasoning": "按主题检索。",
        "arguments": {"query": "retrieval augmented generation"},
    },
}


class PaperSearchModelTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(tracing_context(enabled=False))
        directory = self.enterContext(TemporaryDirectory())
        self.store = LocalArtifactStore(directory)

    def model(self, *responses):
        model = ChatOpenAI(model="injected-model", api_key="test-key")
        generate = self.enterContext(patch.object(model, "_agenerate", new=AsyncMock(side_effect=[
            ChatResult(generations=[ChatGeneration(message=AIMessage(content=json.dumps(response)))])
            for response in responses
        ])))
        return model, generate

    async def test_planning_uses_raw_model_and_repairs_invalid_json(self):
        model, generate = self.model({"query_understanding": {}}, PLAN)
        node = PlanSearchNode(model=model)
        update = await node({"original_prompt": "检索 RAG 方法论文"})

        self.assertIs(node.model, model)
        self.assertIs(node.invoker.model, model)
        self.assertEqual(update["stage"], "searching")
        self.assertEqual(update["query_understanding"]["topic"], UNDERSTANDING["topic"])
        self.assertEqual(generate.await_count, 2)
        messages = generate.call_args.args[0]
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIn("query_understanding", messages[1].content)
        self.assertIn("校验错误", messages[-1].content)

    async def test_invalid_output_stops_after_existing_retry_limit(self):
        model, generate = self.model(*([{}] * 4))
        invoker = ValidatedJsonInvoker(model=model, schema=PromptUnderstandingArgs)
        with self.assertRaises(StructuredOutputError) as error:
            await invoker.ainvoke([HumanMessage(content="检索 RAG")])
        self.assertEqual(error.exception.attempts, 4)
        self.assertEqual(generate.await_count, 4)

    async def test_other_llm_nodes_parse_their_own_schemas(self):
        model, generate = self.model(
            {"action": "finish", "required_additional_count": 0, "reasoning": "结束检索。"},
            {"standard_name": "Journal of Testing"},
            {"paper_abstract": "Source abstract", "keywords": ["RAG"], "ai_abstract": "中文摘要。"},
            {"recommendation_stars": 4, "recommendation_reason": "研究主题匹配。"},
        )
        review = ReviewNode(artifact_store=self.store, model=model)
        with patch("app.llm.graph.workflows.paper_search.nodes.review.settings.PAPER_SEARCH_TARGET_PAPER_COUNT", 5):
            update = await review({"child_run_id": "child-model-test", "progress": {}})
        self.assertEqual(update["stage"], "enriching")

        venue = VenueResolutionNode(
            artifact_store=self.store, model=model,
            cache_store=VenueCacheStore(Path(self.store.base_dir) / "venues.json"),
            easy_scholar_handler=AsyncMock(return_value={"ok": False}),
        )
        resolved, source = await venue.resolve_venue(title="A paper", venue_candidates=["J Test"])
        self.assertEqual((resolved.standard_name, source), ("Journal of Testing", "model_fallback"))

        abstract = AbstractEnrichNode(artifact_store=self.store, model=model)
        enriched = await abstract._enrich_from_pdf_front_pages("PDF text", None, None)
        self.assertEqual(enriched.ai_abstract, "中文摘要。")

        recommend = RecommendationNode(artifact_store=self.store, model=model)
        result = await recommend._recommend_one(
            query_understanding=PromptUnderstandingArgs.model_validate(UNDERSTANDING),
            paper={"paper_info": {"title": "A paper"}},
        )
        self.assertEqual(result.recommendation_stars, 4)
        self.assertEqual(generate.await_count, 4)
        for node in (review, venue, abstract, recommend):
            self.assertIs(node.model, model)

    async def test_review_keeps_pagination_fallback_on_invalid_model_output(self):
        model, generate = self.model(*([{}] * 4))
        node = ReviewNode(artifact_store=self.store, model=model)
        with patch("app.llm.graph.workflows.paper_search.nodes.review.settings.PAPER_SEARCH_TARGET_PAPER_COUNT", 5):
            update = await node({
                "child_run_id": "child-fallback", "progress": {},
                "source_search_stats": {"Google Scholar": {"has_next": False}},
            })
        self.assertEqual(update["stage"], "enriching")
        self.assertEqual(update["search_review_decision"]["reasoning"], "使用确定性兜底策略。")
        self.assertEqual(generate.await_count, 4)

    def test_workflow_binds_all_five_schemas_to_supplied_model(self):
        model, generate = self.model()
        with patch.object(ChatOpenAI, "with_structured_output", autospec=True, side_effect=ChatOpenAI.with_structured_output) as structured:
            graph = build_paper_search_workflow(model=model).get_graph()
        self.assertEqual(structured.call_count, 5)
        self.assertTrue(all(call.args[0] is model for call in structured.call_args_list))
        self.assertEqual({call.args[1].__name__ for call in structured.call_args_list}, {
            "SearchPlanningArgs", "SearchReviewDecision", "ModelVenueInfoDraft",
            "PdfFrontPageEnrichmentResult", "PaperRecommendationResult",
        })
        self.assertIn("search", {edge.target for edge in graph.edges if edge.source == "review"})
        self.assertIn("enrich", {edge.target for edge in graph.edges if edge.source == "review"})
        generate.assert_not_awaited()

    async def test_registered_tool_passes_external_model_without_exposing_it(self):
        model, generate = self.model()
        selected = next(tool for tool in build_tools(None, user_id=1, model=model).get_tools() if tool.name == "paper_search_agent")
        properties = convert_to_openai_tool(selected)["function"]["parameters"]["properties"]
        self.assertEqual(set(properties), {"prompt", "task_id"})
        builder = StateGraph(MessagesState, context_schema=BaseAgentContext)
        builder.add_node("tool", ToolNode([selected]))
        builder.add_edge(START, "tool")
        builder.add_edge("tool", END)
        invoke = AsyncMock(return_value={"content": "检索完成。", "status": "success"})
        target = "app.llm.graph.workflows.paper_search.workflow.build_paper_search_workflow"
        with patch(target) as build:
            build.return_value.ainvoke = invoke
            output = await builder.compile().ainvoke(
                {"messages": [AIMessage(content="", tool_calls=[{
                    "name": selected.name, "args": {"prompt": "检索 RAG", "task_id": 24}, "id": "search-call",
                }])]},
                config={"configurable": {"thread_id": "parent-thread"}},
                context=BaseAgentContext(thread_id="parent-thread", run_id="parent-run", user_id="1"),
            )
        build.assert_called_once_with(model=model)
        self.assertEqual(invoke.call_args.args[0]["task_id"], 24)
        self.assertEqual(output["messages"][-1].content, "检索完成。")
        self.assertEqual(output["messages"][-1].tool_call_id, "search-call")
        generate.assert_not_awaited()
