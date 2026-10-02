from __future__ import annotations

import unittest
from datetime import date
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from langgraph.errors import GraphInterrupt
from pydantic import ValidationError

from app.events.adapter import event_data, sse_name_for
from app.events.bus import EventBus
from app.events.context import EventContext, bind_event_context
from app.llm.graph.workflows.paper_search.nodes.enrich import (
    PaperEnrichmentNode,
)
from app.llm.graph.workflows.paper_search.nodes.plan_search import PlanSearchNode
from app.llm.graph.workflows.paper_search.nodes.review import ReviewNode
from app.llm.graph.workflows.paper_search.nodes.search import (
    SEARCH_HANDLERS,
    SearchNode,
)
from app.llm.graph.workflows.paper_search.progress import (
    instrument_paper_search_node,
)
from app.llm.graph.workflows.paper_search.workflow import (
    build_paper_search_workflow,
)
from app.llm.graph.workflows.paper_search_schemas import SearchTagArgs
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.subagents.paper_search import PaperSearchDelegation
from app.llm.tools.registry import build_tool_registry


class _PlanningModel:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, _messages: list) -> dict:
        self.calls += 1
        return {
            "query_understanding": {
                "topic": "retrieval augmented generation",
                "intent": "method",
                "reasoning": "用户请求方法论文。",
            },
            "query_plan": {
                "display_query": "retrieval augmented generation method",
                "reasoning": "由主题和方法意图生成检索式。",
                "arguments": {
                    "query": "retrieval augmented generation method",
                },
            },
        }


class _FailingPlanningModel:
    async def ainvoke(self, _messages: list) -> dict:
        raise RuntimeError("model unavailable")


class _InvalidPlanningModel:
    async def ainvoke(self, _messages: list) -> dict:
        return {
            "query_understanding": {
                "topic": "retrieval augmented generation",
                "intent": "method",
                "reasoning": "用户请求方法论文。",
            },
            "query_plan": {
                "display_query": "retrieval augmented generation method",
                "reasoning": "由主题和方法意图生成检索式。",
                "arguments": {"query": ""},
            },
        }


class _ReviewModel:
    def __init__(self, result: dict | Exception) -> None:
        self.result = result
        self.calls = 0

    async def ainvoke(self, _messages: list) -> dict:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _FailingArtifactStore:
    async def write_json(self, **_kwargs):
        raise RuntimeError("artifact store unavailable")


def _search_state(*, progress: dict | None = None) -> dict:
    return {
        "run_id": "single-source-search",
        "child_run_id": "child-single-source-search",
        "warnings": [],
        "progress": progress or {},
        "active_search_sources": ["Google Scholar"],
        "active_source_query_plans": [{
            "source": "Google Scholar",
            "display_query": "retrieval augmented generation",
            "reasoning": "测试检索计划。",
            "arguments": {"query": "retrieval augmented generation", "num": 10},
        }],
        "raw_result_artifact_refs": [],
    }


class PaperSearchSingleSourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_plan_search_writes_intent_tag_and_plan_in_one_call(self) -> None:
        model = _PlanningModel()
        result = await PlanSearchNode(
            model=model,
        )(
            {
                "original_prompt": "检索 RAG 方法论文",
                "warnings": [],
            }
        )

        self.assertEqual(model.calls, 1)
        self.assertEqual(result["stage"], "searching")
        self.assertEqual(result["status"], "running")
        self.assertEqual(
            result["query_understanding"]["topic"],
            "retrieval augmented generation",
        )
        self.assertEqual(result["search_tag"], {
            "yearTag": 0,
            "paperTag": [1, 2],
            "sourceTag": ["Google Scholar"],
        })
        self.assertEqual(result["source_query_plans"][0]["source"], "Google Scholar")
        self.assertEqual(
            result["source_query_plans"][0]["arguments"]["query"],
            "retrieval augmented generation method",
        )

    async def test_plan_search_uses_default_types_and_pagination(self) -> None:
        result = await PlanSearchNode(
            model=_PlanningModel(),
            pagination_settings={
                "google_max_pages": 2,
                "google_page_size": 10,
                "google_total_limit": 20,
            },
        )({"original_prompt": "检索 RAG 方法论文"})

        self.assertEqual(result["search_tag"]["yearTag"], 0)
        self.assertEqual(result["search_tag"]["paperTag"], [1, 2])
        self.assertEqual(
            result["source_query_plans"][0]["arguments"],
            {
                "query": "retrieval augmented generation method",
                "start": 0,
                "num": 10,
                "total_limit": 20,
                "max_pages": 2,
                "year_from": 2000,
                "year_to": date.today().year,
                "review_only": False,
            },
        )

    async def test_plan_search_uses_explicit_intent_years(self) -> None:
        class _DatedPlanningModel:
            async def ainvoke(self, _messages: list) -> dict:
                return {
                    "query_understanding": {
                        "topic": "retrieval augmented generation",
                        "intent": "method",
                        "yearFrom": 2020,
                        "yearTo": 2021,
                        "reasoning": "用户明确限定了年份。",
                    },
                    "query_plan": {
                        "display_query": "retrieval augmented generation",
                        "reasoning": "使用主题构建检索式。",
                        "arguments": {
                            "query": "retrieval augmented generation",
                        },
                    },
                }

        result = await PlanSearchNode(
            model=_DatedPlanningModel(),
            pagination_settings={
                "google_max_pages": 1,
                "google_page_size": 10,
                "google_total_limit": 10,
            },
        )({"original_prompt": "检索 2020 至 2021 年 RAG 方法论文"})

        arguments = result["source_query_plans"][0]["arguments"]
        self.assertEqual(arguments["year_from"], 2020)
        self.assertEqual(arguments["year_to"], 2021)

    async def test_plan_search_fails_when_model_output_is_unavailable(self) -> None:
        result = await PlanSearchNode(model=_FailingPlanningModel())(
            {"original_prompt": "检索 RAG 方法论文"}
        )

        self.assertEqual(result["stage"], "failed")
        self.assertEqual(result["status"], "failed")
        self.assertIn("model unavailable", result["intent_error"])

    async def test_plan_search_fails_for_invalid_query_arguments(self) -> None:
        result = await PlanSearchNode(model=_InvalidPlanningModel())(
            {"original_prompt": "检索 RAG 方法论文"}
        )

        self.assertEqual(result["stage"], "failed")
        self.assertIn("query", result["intent_error"])
        self.assertEqual(result["source_query_plans"], [])

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
                    "child_run_id": "child-single-source-test",
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

    async def test_search_updates_stats_cursor_and_artifact_in_one_node(self) -> None:
        async def handler(_arguments: dict, _session) -> dict:
            return {
                "ok": True,
                "query": "retrieval augmented generation",
                "search_query": "q=rag",
                "papers": [{"title": "A paper"}],
                "metadata": {
                    "total_results": 5,
                    "pagination": {
                        "next_offset": 10,
                        "has_next": True,
                        "pages_fetched": 1,
                    },
                },
            }

        async def save_artifact(**_kwargs):
            return SimpleNamespace(artifact_uri="artifact://search-result")

        with (
            patch.dict(SEARCH_HANDLERS, {"Google Scholar": handler}),
            patch(
                "app.llm.graph.workflows.paper_search.nodes.search."
                "save_paper_search_artifact",
                new=save_artifact,
            ),
        ):
            result = await SearchNode()(_search_state(), {})

        self.assertEqual(result["stage"], "normalizing")
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["progress"]["discovered"], 1)
        self.assertEqual(
            result["source_search_cursors"]["Google Scholar"]["next_offset"],
            10,
        )
        self.assertEqual(
            result["source_search_stats"]["Google Scholar"]["searched_pages"],
            1,
        )
        self.assertEqual(result["raw_result_artifact_refs"], ["artifact://search-result"])

    async def test_search_does_not_require_parent_run_id(self) -> None:
        state = _search_state()
        state.pop("run_id")
        state["active_search_sources"] = []

        result = await SearchNode()(state, {})

        self.assertEqual(result["stage"], "normalizing")
        self.assertEqual(result["status"], "running")

    async def test_supplemental_search_failure_with_candidates_stays_running(self) -> None:
        async def handler(_arguments: dict, _session) -> dict:
            return {
                "ok": False,
                "error": "Google Scholar unavailable",
                "papers": [],
                "metadata": {"pagination": {}},
            }

        async def save_artifact(**_kwargs):
            return SimpleNamespace(artifact_uri="artifact://failed-search")

        with (
            patch.dict(SEARCH_HANDLERS, {"Google Scholar": handler}),
            patch(
                "app.llm.graph.workflows.paper_search.nodes.search."
                "save_paper_search_artifact",
                new=save_artifact,
            ),
        ):
            result = await SearchNode()(
                _search_state(progress={"new_candidates": 2}),
                {},
            )

        self.assertEqual(result["stage"], "normalizing")
        self.assertEqual(result["status"], "running")
        self.assertTrue(result["degraded"])
        self.assertEqual(
            result["source_summaries"]["Google Scholar"]["status"],
            "failed",
        )
        self.assertIn("Google Scholar unavailable", result["warnings"])

    async def test_initial_search_failure_without_candidates_fails(self) -> None:
        async def handler(_arguments: dict, _session) -> dict:
            return {
                "ok": False,
                "error": "Google Scholar unavailable",
                "papers": [],
                "metadata": {"pagination": {}},
            }

        async def save_artifact(**_kwargs):
            return SimpleNamespace(artifact_uri="artifact://initial-failure")

        with (
            patch.dict(SEARCH_HANDLERS, {"Google Scholar": handler}),
            patch(
                "app.llm.graph.workflows.paper_search.nodes.search."
                "save_paper_search_artifact",
                new=save_artifact,
            ),
        ):
            result = await SearchNode()(_search_state(), {})

        self.assertEqual(result["stage"], "failed")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "所有检索来源均失败。")

    async def test_partial_search_result_keeps_partial_failed_status(self) -> None:
        async def handler(_arguments: dict, _session) -> dict:
            return {
                "ok": True,
                "papers": [{"title": "A paper"}],
                "warnings": ["next page unavailable"],
                "metadata": {"pagination": {}},
            }

        async def save_artifact(**_kwargs):
            return SimpleNamespace(artifact_uri="artifact://partial-search")

        with (
            patch.dict(SEARCH_HANDLERS, {"Google Scholar": handler}),
            patch(
                "app.llm.graph.workflows.paper_search.nodes.search."
                "save_paper_search_artifact",
                new=save_artifact,
            ),
        ):
            result = await SearchNode()(_search_state(), {})

        self.assertEqual(result["stage"], "normalizing")
        self.assertEqual(result["status"], "partial_failed")
        self.assertTrue(result["degraded"])

    async def test_review_skips_model_at_target_and_enters_enrichment(self) -> None:
        model = _ReviewModel(RuntimeError("model should not run"))
        with TemporaryDirectory() as directory:
            result = await ReviewNode(
                model=model,
                artifact_store=LocalArtifactStore(directory),
            )({
                "run_id": "review-target",
                "child_run_id": "child-review-target",
                "progress": {"new_candidates": 999},
                "source_search_stats": {},
            })

        self.assertEqual(model.calls, 0)
        self.assertEqual(result["stage"], "enriching")
        self.assertEqual(result["status"], "running")
        self.assertIn("search_review_decision_artifact_ref", result)

    async def test_review_generates_next_page_plan_and_increments_round(self) -> None:
        model = _ReviewModel({
            "action": "supplement",
            "required_additional_count": 1,
            "source_actions": [{
                "source": "Google Scholar",
                "strategy": "next_page",
                "requested_pages": 2,
                "priority": 1,
                "reason": "继续获取候选论文。",
            }],
            "reasoning": "候选论文不足。",
        })
        with TemporaryDirectory() as directory:
            result = await ReviewNode(
                model=model,
                artifact_store=LocalArtifactStore(directory),
            )({
                "run_id": "review-supplement",
                "child_run_id": "child-review-supplement",
                "progress": {},
                "source_search_stats": {"Google Scholar": {"has_next": True}},
                "source_search_cursors": {
                    "Google Scholar": {"has_next": True, "next_offset": 20},
                },
                "source_query_plans": [{
                    "source": "Google Scholar",
                    "arguments": {"query": "rag", "num": 10, "start": 0},
                }],
                "supplemental_search_round": 0,
            })

        plan = result["active_source_query_plans"][0]
        self.assertEqual(result["stage"], "searching")
        self.assertEqual(result["supplemental_search_round"], 1)
        self.assertEqual(plan["arguments"]["start"], 20)
        self.assertEqual(plan["arguments"]["max_pages"], 2)
        self.assertEqual(plan["arguments"]["total_limit"], 20)

    async def test_review_uses_pagination_fallback_when_model_fails(self) -> None:
        model = _ReviewModel(RuntimeError("model unavailable"))
        with TemporaryDirectory() as directory:
            result = await ReviewNode(
                model=model,
                artifact_store=LocalArtifactStore(directory),
            )({
                "run_id": "review-fallback",
                "child_run_id": "child-review-fallback",
                "progress": {},
                "source_search_stats": {"Google Scholar": {"has_next": True}},
                "source_search_cursors": {
                    "Google Scholar": {"has_next": True, "next_offset": 10},
                },
                "source_query_plans": [{
                    "source": "Google Scholar",
                    "arguments": {"query": "rag", "num": 10},
                }],
            })

        self.assertEqual(result["stage"], "searching")
        self.assertEqual(
            result["active_source_query_plans"][0]["arguments"]["max_pages"],
            1,
        )

    async def test_review_marks_unavailable_supplement_as_degraded(self) -> None:
        model = _ReviewModel({
            "action": "supplement",
            "required_additional_count": 1,
            "source_actions": [{
                "source": "Google Scholar",
                "strategy": "next_page",
                "requested_pages": 1,
                "priority": 1,
                "reason": "继续获取候选论文。",
            }],
            "reasoning": "候选论文不足。",
        })
        with TemporaryDirectory() as directory:
            result = await ReviewNode(
                model=model,
                artifact_store=LocalArtifactStore(directory),
            )({
                "run_id": "review-unavailable",
                "child_run_id": "child-review-unavailable",
                "progress": {},
                "source_search_stats": {},
                "source_search_cursors": {
                    "Google Scholar": {"has_next": False},
                },
                "source_query_plans": [],
                "warnings": [],
            })

        self.assertEqual(result["stage"], "enriching")
        self.assertEqual(result["status"], "partial_failed")
        self.assertTrue(result["degraded"])

    async def test_review_fails_when_decision_artifact_cannot_be_saved(self) -> None:
        result = await ReviewNode(
            model=_ReviewModel({
                "action": "finish",
                "required_additional_count": 0,
                "reasoning": "结束检索。",
            }),
            artifact_store=_FailingArtifactStore(),
        )({
            "run_id": "review-artifact-failure",
            "child_run_id": "child-review-artifact-failure",
            "progress": {"new_candidates": 999},
            "source_search_stats": {},
        })

        self.assertEqual(result["stage"], "failed")
        self.assertIn("artifact store unavailable", result["error"])

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
            node_name="search",
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
            node_name="search",
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
            node_name="search",
            node=node,
        )
        with bind_event_context(EventContext(bus=bus, run_id="run-1")):
            with self.assertRaises(GraphInterrupt):
                await wrapped({}, {})

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["data"]["phase_state"], "started")


class PaperSearchSingleSourceValidationTests(unittest.TestCase):
    def test_workflow_has_search_and_review_nodes(self) -> None:
        nodes = build_paper_search_workflow().get_graph().nodes
        self.assertIn("plan_search", nodes)
        self.assertIn("finalize_pdfs", nodes)
        self.assertNotIn("prepare_search", nodes)
        self.assertNotIn("generate_queries", nodes)
        self.assertNotIn("save_pdfs_to_oss", nodes)
        self.assertNotIn("cleanup_downloaded_pdfs", nodes)
        self.assertNotIn("intent_understanding", nodes)
        self.assertNotIn("build_search_tag", nodes)
        self.assertIn("search", nodes)
        self.assertIn("review", nodes)
        self.assertNotIn("confirm", nodes)
        self.assertNotIn("search_google", nodes)
        self.assertNotIn("finalize_source_search", nodes)
        self.assertNotIn("search_review", nodes)
        self.assertNotIn("supplemental_search", nodes)

    def test_plan_search_routes_directly_to_search(self) -> None:
        graph = build_paper_search_workflow().get_graph()
        targets = {
            edge.target
            for edge in graph.edges
            if edge.source == "plan_search"
        }

        self.assertIn("search", targets)
        self.assertNotIn("confirm", targets)

    def test_review_routes_to_search_or_enrichment(self) -> None:
        graph = build_paper_search_workflow().get_graph()
        review_targets = {
            edge.target for edge in graph.edges if edge.source == "review"
        }

        self.assertIn("search", review_targets)
        self.assertIn("enrich", review_targets)

    def test_arxiv_and_dblp_are_rejected_for_search_tags(self) -> None:
        for source in ("arXiv", "DBLP"):
            with self.assertRaises(ValidationError):
                SearchTagArgs.model_validate({"sourceTag": [source]})

    def test_delegation_schema_does_not_expose_constraints(self) -> None:
        schema = PaperSearchDelegation.model_json_schema()
        self.assertNotIn("constraints", schema["properties"])

    def test_subgraph_only_uses_google_scholar(self) -> None:
        names = {schema["name"] for schema in build_tool_registry().schemas()}
        self.assertIn("google_scholar_search", names)
        self.assertIn("arxiv_search", names)
        self.assertIn("dblp_search", names)
        self.assertEqual(set(SEARCH_HANDLERS), {"Google Scholar"})


if __name__ == "__main__":
    unittest.main()
