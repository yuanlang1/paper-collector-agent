from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

from langgraph.errors import GraphInterrupt
from pydantic import ValidationError

from app.events.adapter import event_data, sse_name_for
from app.events.bus import EventBus
from app.events.context import EventContext, bind_event_context
from app.llm.graph.workflows.paper_search.nodes.generate_queries import (
    BuildSourceQueryPlanNode,
)
from app.llm.graph.workflows.paper_search.nodes.enrich import (
    PaperEnrichmentNode,
)
from app.llm.graph.workflows.paper_search.nodes.search import SEARCH_HANDLERS
from app.llm.graph.workflows.paper_search.progress import (
    instrument_paper_search_node,
)
from app.llm.graph.workflows.paper_search.workflow import (
    build_paper_search_workflow,
)
from app.llm.graph.workflows.paper_search_schemas import SearchTagArgs
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.subagents.paper_search import PaperSearchConstraints
from app.llm.tools.registry import build_tool_registry


class _SourcePlanModel:
    async def ainvoke(self, _messages: list) -> dict:
        return {
            "plans": [
                {
                    "source": "Google Scholar",
                    "display_query": "retrieval augmented generation",
                    "reasoning": "由主题和关键词生成检索式。",
                    "arguments": {"query": "retrieval augmented generation"},
                }
            ]
        }


class PaperSearchSingleSourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_query_plan_is_always_google_scholar(self) -> None:
        result = await BuildSourceQueryPlanNode(
            model=_SourcePlanModel(),
            pagination_settings={
                "google_max_pages": 2,
                "google_page_size": 10,
                "google_total_limit": 20,
            },
        )(
            {
                "query_understanding": {
                    "topic": "retrieval augmented generation",
                    "intent": "method",
                    "reasoning": "用户请求方法论文。",
                },
                "search_tag": {"sourceTag": ["Google Scholar"]},
            },
        )

        self.assertEqual(result["requested_sources"], ["Google Scholar"])
        self.assertEqual(
            result["source_query_plans"][0]["source"],
            "Google Scholar",
        )
        self.assertEqual(
            result["source_query_plans"][0]["arguments"]["num"],
            10,
        )

    async def test_enrichment_keeps_google_scholar_pdf_link(self) -> None:
        with TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            manifest = await store.write_json(
                run_id="single-source-test",
                step_key="normalize",
                source="Google Scholar",
                kind="normalized_manifest_json",
                payload={
                    "papers": [
                        {
                            "paper_info": {
                                "title": "A paper",
                                "pdf_url": "https://arxiv.org/pdf/1234.5678",
                            },
                        },
                    ],
                },
            )

            update = await PaperEnrichmentNode(artifact_store=store)(
                {
                    "run_id": "single-source-test",
                    "normalized_manifest_artifact_ref": manifest.artifact_uri,
                },
            )
            enriched = await store.read_json_uri(
                update["enrichment_manifest_artifact_ref"],
            )

        self.assertEqual(len(enriched["papers"]), 1)
        self.assertEqual(enriched["removed_without_pdf"], [])
        self.assertEqual(update["progress"]["pdf_enriched"], 0)
        self.assertEqual(update["progress"]["venue_enriched"], 0)

    async def test_progress_events_keep_outer_delegation_scope(self) -> None:
        events: list[dict] = []
        outer_scope = {
            "action_id": "action-1",
            "delegation_id": "delegation-1",
            "subagent": "paper_search_agent",
            "workflow": "paper_search",
        }

        async def node(_state: dict) -> dict:
            return {
                "progress": {"discovered": 2},
                "source_search_stats": {
                    "Google Scholar": {"discovered_count": 2},
                },
            }

        bus = EventBus()

        async def collect(event) -> None:
            events.append({"event": sse_name_for(event), **event_data(event)})

        bus.subscribe(collect)

        wrapped = instrument_paper_search_node(
            node_name="search_google",
            node=node,
        )
        with bind_event_context(
            EventContext(bus=bus, run_id="run-1", scope=outer_scope),
        ):
            await wrapped({}, {})

        self.assertEqual([event["event"] for event in events], [
            "subagent_progress",
            "subagent_progress",
        ])
        self.assertTrue(all("timeline_step" not in event for event in events))
        self.assertTrue(all(
            event["delegation_id"] == "delegation-1" for event in events
        ))
        self.assertEqual(events[0]["data"]["phase_state"], "started")
        self.assertEqual(events[1]["data"]["phase_state"], "completed")
        self.assertEqual(events[1]["data"]["counts"], {"discovered": 2})
        self.assertTrue(all("message" not in event for event in events))
        self.assertTrue(all("iteration" not in event for event in events))

    async def test_progress_exception_is_reported_and_propagated(self) -> None:
        events: list[dict] = []

        async def node(_state: dict) -> dict:
            raise RuntimeError("search failed")

        bus = EventBus()

        async def collect(event) -> None:
            events.append({"event": sse_name_for(event), **event_data(event)})

        bus.subscribe(collect)
        wrapped = instrument_paper_search_node(
            node_name="search_google",
            node=node,
        )
        with bind_event_context(EventContext(bus=bus, run_id="run-1")):
            with self.assertRaisesRegex(RuntimeError, "search failed"):
                await wrapped({}, {})

        self.assertEqual(events[-1]["data"]["phase_state"], "failed")
        self.assertEqual(events[-1]["data"]["error"], "search failed")

    async def test_graph_interrupt_is_propagated_without_failure_event(self) -> None:
        events: list[dict] = []

        async def node(_state: dict) -> dict:
            raise GraphInterrupt()

        bus = EventBus()

        async def collect(event) -> None:
            events.append({"event": sse_name_for(event), **event_data(event)})

        bus.subscribe(collect)
        wrapped = instrument_paper_search_node(
            node_name="search_google",
            node=node,
        )
        with bind_event_context(EventContext(bus=bus, run_id="run-1")):
            with self.assertRaises(GraphInterrupt):
                await wrapped({}, {})

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["data"]["phase_state"], "started")


class PaperSearchSingleSourceValidationTests(unittest.TestCase):
    def test_workflow_has_only_google_scholar_search_node(self) -> None:
        nodes = build_paper_search_workflow(skip_confirmation=True).get_graph().nodes
        self.assertIn("search_google", nodes)
        self.assertNotIn("search_arxiv", nodes)
        self.assertNotIn("search_dblp", nodes)

    def test_arxiv_and_dblp_are_rejected_for_new_delegations(self) -> None:
        for source in ("arXiv", "DBLP"):
            with self.assertRaises(ValidationError):
                SearchTagArgs.model_validate({"sourceTag": [source]})
            with self.assertRaises(ValidationError):
                PaperSearchConstraints.model_validate({"sources": [source]})

    def test_subgraph_only_uses_google_scholar(self) -> None:
        names = {schema["name"] for schema in build_tool_registry().schemas()}
        self.assertIn("google_scholar_search", names)
        self.assertIn("arxiv_search", names)
        self.assertIn("dblp_search", names)
        self.assertEqual(set(SEARCH_HANDLERS), {"Google Scholar"})


if __name__ == "__main__":
    unittest.main()
