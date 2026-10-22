from __future__ import annotations

import asyncio
import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphInterrupt
from pydantic import ValidationError

from app.events.adapter import event_data, sse_name_for
from app.events.bus import EventBus
from app.events.context import EventContext, bind_event_context
from app.infrastructure.grpc.task_service_grpc_client import TaskState
from app.llm.graph.workflows.paper_search.nodes.enrich import (
    PaperEnrichmentNode,
)
from app.llm.graph.workflows.paper_search.nodes.dowload_pdf import PdfDownloadNode
from app.llm.graph.workflows.paper_search.nodes.crossref_enrich import (
    CrossrefMetadataEnrichmentNode,
)
from app.llm.graph.workflows.paper_search.nodes.create_task import (
    CreatePaperSearchTaskNode,
)
from app.llm.graph.workflows.paper_search.nodes.initialize import (
    initialize_paper_search_node,
)
from app.llm.graph.workflows.paper_search.nodes.update_task_status import (
    UpdatePaperSearchTaskStatusNode,
)
from app.llm.graph.workflows.paper_search.nodes.plan_search import PlanSearchNode
from app.llm.graph.workflows.paper_search.nodes.review import ReviewNode
from app.llm.graph.workflows.paper_search.nodes.recommend import (
    PaperRecommendationResult,
    RecommendationNode,
)
from app.llm.graph.workflows.paper_search.nodes.search import (
    SEARCH_HANDLERS,
    SearchNode,
)
from app.llm.graph.workflows.paper_search.progress import (
    instrument_paper_search_node,
)
from app.llm.graph.workflows.paper_search.nodes.venue import (
    VenueCacheStore,
    VenueResolutionNode,
)
from app.llm.graph.workflows.paper_search.state import PaperSearchWorkflowState
from app.llm.graph.workflows.paper_search.workflow import (
    build_paper_search_workflow,
)
from app.llm.graph.workflows.paper_search_schemas import SearchTagArgs
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.subagents.paper_search import PaperSearchParams
from app.llm.tools.registry import build_tool_registry


import httpx

from app.infrastructure.grpc.paper_service_grpc_client import PaperServiceGrpcClient
from app.llm.graph.workflows.paper_search.nodes.filter import (
    NormalizeDeduplicateFilterNode, _merge_envelope, _to_paper_info_draft,
)
from app.llm.graph.workflows.paper_search.nodes.persist import PersistRecommendedPapersNode
from app.llm.graph.workflows.paper_search.paper_cache import PaperCacheStore
from app.llm.tools.search_tools.crossref.search_crossref import _published_date
from app.llm.tools.search_tools.google_scholar.search_google_scholar import (
    _parse_google_scholar_item, google_scholar_search_service, SerpApiSearchError,
)
from tests.chat_model_stub import as_chat_model


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


def _concurrent_paper(index: int, *, doi: str | None = None) -> dict:
    paper_info = {
        "title": f"Paper {index}",
        "authors": "Ada Lovelace",
        "paper_abstract": "Abstract",
    }
    if doi:
        paper_info["doi"] = doi
    return {"paper_info": paper_info}


_RECOMMENDATION_QUERY_UNDERSTANDING = {
    "topic": "retrieval augmented generation",
    "intent": "method",
    "reasoning": "用户要求检索检索增强生成方法论文。",
}


class PaperSearchSingleSourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_binds_provided_task_id(self) -> None:
        update = await initialize_paper_search_node(
            {
                "prompt": "检索 RAG 方法论文",
                "task_id": 24,
            }
        )

        self.assertEqual(update["paper_service_task_id"], 24)
        self.assertEqual(update["stage"], "preparing_search")
        self.assertEqual(update["status"], "running")

    async def test_initialize_clears_task_id_when_none_is_provided(self) -> None:
        update = await initialize_paper_search_node(
            {"prompt": "检索 RAG 方法论文"}
        )

        self.assertIsNone(update["paper_service_task_id"])

    async def test_existing_task_is_marked_running_before_planning(self) -> None:
        client = Mock(update_task_status=AsyncMock(return_value={"ok": True}))
        update = await UpdatePaperSearchTaskStatusNode(client=client)(
            {
                "paper_service_task_id": 24,
                "stage": "preparing_search",
                "status": "running",
            }
        )

        client.update_task_status.assert_awaited_once_with(
            task_id=24,
            task_state=TaskState.SEARCH_RUNNING,
            error_message=None,
        )
        self.assertEqual(update["remote_task_state"], "SEARCH_RUNNING")

    async def test_running_status_update_failure_stops_search(self) -> None:
        client = Mock(
            update_task_status=AsyncMock(
                return_value={"ok": False, "error": "offline"}
            )
        )
        update = await UpdatePaperSearchTaskStatusNode(client=client)(
            {
                "paper_service_task_id": 24,
                "stage": "preparing_search",
                "status": "running",
                "warnings": [],
                "progress": {},
            }
        )

        self.assertEqual(update["stage"], "failed")
        self.assertIn("offline", update["error"])

    async def test_plan_search_writes_intent_tag_and_plan_in_one_call(self) -> None:
        model = _PlanningModel()
        result = await PlanSearchNode(
            model=as_chat_model(model),
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
            model=as_chat_model(_PlanningModel()),
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
            model=as_chat_model(_DatedPlanningModel()),
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
        result = await PlanSearchNode(model=as_chat_model(_FailingPlanningModel()))(
            {"original_prompt": "检索 RAG 方法论文"}
        )

        self.assertEqual(result["stage"], "failed")
        self.assertEqual(result["status"], "failed")
        self.assertIn("model unavailable", result["intent_error"])

    async def test_plan_search_fails_for_invalid_query_arguments(self) -> None:
        result = await PlanSearchNode(model=as_chat_model(_InvalidPlanningModel()))(
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
        self.assertNotIn("degraded", result)
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

    async def test_partial_search_result_continues_as_running(self) -> None:
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
        self.assertEqual(result["status"], "running")
        self.assertNotIn("degraded", result)

    async def test_review_skips_model_at_target_and_enters_enrichment(self) -> None:
        model = _ReviewModel(RuntimeError("model should not run"))
        with TemporaryDirectory() as directory:
            result = await ReviewNode(
                model=as_chat_model(model),
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
                model=as_chat_model(model),
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
                model=as_chat_model(model),
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

    async def test_review_warns_when_supplement_is_unavailable(self) -> None:
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
                model=as_chat_model(model),
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
        self.assertEqual(result["status"], "running")
        self.assertNotIn("degraded", result)
        self.assertIn("无可用补充检索来源，继续后续流程。", result["warnings"])

    async def test_review_fails_when_decision_artifact_cannot_be_saved(self) -> None:
        result = await ReviewNode(
            model=as_chat_model(_ReviewModel({
                "action": "finish",
                "required_additional_count": 0,
                "reasoning": "结束检索。",
            })),
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

    async def test_pdf_download_uses_raw_dict_handler(self) -> None:
        with TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            child_run_id = "child-tool-result-pdf"
            downloaded_path = store.base_dir / "papers" / "paper.pdf"
            downloaded_path.parent.mkdir(parents=True)
            downloaded_path.write_bytes(b"%PDF-1.4")
            input_artifact = await store.write_json(
                run_id=child_run_id,
                step_key="resolve_venues",
                source="venue",
                kind="paper_info_venue_manifest_json",
                payload={
                    "papers": [
                        {
                            "paper_info": {
                                "title": "A Paper",
                                "authors": "Ada Lovelace",
                                "pdf_url": "https://example.test/paper.pdf",
                            }
                        }
                    ]
                },
            )
            existing_artifact = await store.write_json(
                run_id=child_run_id,
                step_key="filter_existing",
                source="filter",
                kind="existing_papers_manifest_json",
                payload={"papers": [{"paper_id": 1}]},
            )
            calls: list[dict] = []

            async def fake_download(params: dict, _context: object) -> dict:
                calls.append(params)
                return {
                    "ok": True,
                    "data": {
                        "saved_path": str(downloaded_path),
                        "artifact_uri": store.to_artifact_uri(downloaded_path),
                        "final_url": params["url"],
                        "sha256": "a" * 64,
                        "size_bytes": downloaded_path.stat().st_size,
                    },
                }

            with patch(
                "app.llm.graph.workflows.paper_search.nodes.dowload_pdf."
                "download_file_handler",
                new=fake_download,
            ):
                update = await PdfDownloadNode(artifact_store=store)(
                    {
                        "child_run_id": child_run_id,
                        "venue_manifest_artifact_ref": input_artifact.artifact_uri,
                        "existing_papers_manifest_artifact_ref": (
                            existing_artifact.artifact_uri
                        ),
                    }
                )

            self.assertEqual(update["stage"], "enriching_abstract")
            self.assertEqual(update["downloaded_pdf_paths"], [str(downloaded_path)])
            self.assertEqual(calls[0]["url"], "https://example.test/paper.pdf")
            manifest = await store.read_json_uri(
                update["pdf_manifest_artifact_ref"]
            )
            self.assertEqual(
                manifest["papers"][0]["pdf_download"]["status"],
                "success",
            )

    async def test_pdf_download_continues_when_only_existing_papers_remain(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            child_run_id = "child-existing-papers-only"
            input_artifact = await store.write_json(
                run_id=child_run_id,
                step_key="resolve_venues",
                source="venue",
                kind="paper_info_venue_manifest_json",
                payload={"papers": []},
            )
            existing_artifact = await store.write_json(
                run_id=child_run_id,
                step_key="filter_existing",
                source="filter",
                kind="existing_papers_manifest_json",
                payload={"papers": [{"paper_id": 24}]},
            )

            update = await PdfDownloadNode(artifact_store=store)(
                {
                    "child_run_id": child_run_id,
                    "venue_manifest_artifact_ref": input_artifact.artifact_uri,
                    "existing_papers_manifest_artifact_ref": (
                        existing_artifact.artifact_uri
                    ),
                }
            )

            self.assertEqual(update["stage"], "enriching_abstract")
            self.assertIn("无新增 PDF 可用，继续处理已有论文。", update["warnings"])
            manifest = await store.read_json_uri(
                update["pdf_manifest_artifact_ref"]
            )
            self.assertEqual(manifest["papers"], [])

    async def test_pdf_download_continues_when_new_downloads_fail_but_existing_remain(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            child_run_id = "child-pdf-failure-existing-papers"
            input_artifact = await store.write_json(
                run_id=child_run_id,
                step_key="resolve_venues",
                source="venue",
                kind="paper_info_venue_manifest_json",
                payload={
                    "papers": [
                        {
                            "paper_info": {
                                "title": "New Paper",
                                "authors": "Ada Lovelace",
                                "pdf_url": "https://example.test/new.pdf",
                            }
                        }
                    ]
                },
            )
            existing_artifact = await store.write_json(
                run_id=child_run_id,
                step_key="filter_existing",
                source="filter",
                kind="existing_papers_manifest_json",
                payload={"papers": [{"paper_id": 24}]},
            )

            async def failed_download(_params: dict, _context: object) -> dict:
                return {"ok": False, "error": {"message": "forbidden"}}

            with patch(
                "app.llm.graph.workflows.paper_search.nodes.dowload_pdf."
                "download_file_handler",
                new=failed_download,
            ):
                update = await PdfDownloadNode(artifact_store=store)(
                    {
                        "child_run_id": child_run_id,
                        "venue_manifest_artifact_ref": input_artifact.artifact_uri,
                        "existing_papers_manifest_artifact_ref": (
                            existing_artifact.artifact_uri
                        ),
                    }
                )

            self.assertEqual(update["stage"], "enriching_abstract")
            self.assertEqual(update["progress"]["pdf_downloaded"], 0)
            self.assertEqual(update["progress"]["pdf_download_failed"], 1)
            self.assertIn("无新增 PDF 可用，继续处理已有论文。", update["warnings"])

    async def test_pdf_download_fails_when_no_new_or_existing_papers_remain(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            child_run_id = "child-no-papers"
            input_artifact = await store.write_json(
                run_id=child_run_id,
                step_key="resolve_venues",
                source="venue",
                kind="paper_info_venue_manifest_json",
                payload={"papers": []},
            )
            existing_artifact = await store.write_json(
                run_id=child_run_id,
                step_key="filter_existing",
                source="filter",
                kind="existing_papers_manifest_json",
                payload={"papers": []},
            )

            update = await PdfDownloadNode(artifact_store=store)(
                {
                    "child_run_id": child_run_id,
                    "venue_manifest_artifact_ref": input_artifact.artifact_uri,
                    "existing_papers_manifest_artifact_ref": (
                        existing_artifact.artifact_uri
                    ),
                }
            )

            self.assertEqual(update["stage"], "failed")
            self.assertEqual(update["warnings"][-1], "没有可继续处理的论文。")
            self.assertEqual(update["error"], "没有可继续处理的论文。")

    async def test_pdf_download_limits_concurrency_and_keeps_input_order(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            child_run_id = "child-concurrent-pdf"
            papers = [
                {
                    "paper_info": {
                        "title": f"Paper {index}",
                        "authors": "Ada Lovelace",
                        "pdf_url": f"https://example.test/{index}.pdf",
                    }
                }
                for index in range(5)
            ]
            input_artifact = await store.write_json(
                run_id=child_run_id,
                step_key="resolve_venues",
                source="venue",
                kind="paper_info_venue_manifest_json",
                payload={"papers": papers},
            )
            active = 0
            max_active = 0
            started = asyncio.Event()
            release = asyncio.Event()

            async def fake_download(params: dict, _context: object) -> dict:
                nonlocal active, max_active
                active += 1
                max_active = max(max_active, active)
                if active == 4:
                    started.set()
                try:
                    await release.wait()
                finally:
                    active -= 1
                saved_path = str(store.base_dir / "papers" / params["file_name"])
                return {
                    "ok": True,
                    "data": {
                        "saved_path": saved_path,
                        "artifact_uri": store.to_artifact_uri(Path(saved_path)),
                        "final_url": params["url"],
                        "sha256": "a" * 64,
                        "size_bytes": 1,
                    },
                }

            with patch(
                "app.llm.graph.workflows.paper_search.nodes.dowload_pdf."
                "download_file_handler",
                new=fake_download,
            ):
                task = asyncio.create_task(
                    PdfDownloadNode(artifact_store=store)(
                        {
                            "child_run_id": child_run_id,
                            "venue_manifest_artifact_ref": input_artifact.artifact_uri,
                        }
                    )
                )
                await asyncio.wait_for(started.wait(), timeout=1)
                self.assertEqual(max_active, 4)
                release.set()
                update = await task

            manifest = await store.read_json_uri(update["pdf_manifest_artifact_ref"])
            self.assertEqual(
                [paper["paper_info"]["title"] for paper in manifest["papers"]],
                [f"Paper {index}" for index in range(5)],
            )

    async def test_recommendation_limits_concurrency_and_preserves_results(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            child_run_id = "child-concurrent-recommendation"
            artifact = await store.write_json(
                run_id=child_run_id,
                step_key="abstract_enrichment",
                source="test",
                kind="paper_info_abstract_manifest_json",
                payload={
                    "papers": [
                        _concurrent_paper(index)
                        for index in range(4)
                    ]
                },
            )
            started = asyncio.Event()
            release = asyncio.Event()

            class Model:
                def __init__(self) -> None:
                    self.active = 0
                    self.max_active = 0

                async def ainvoke(self, messages: list):
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                    if self.active == 3:
                        started.set()
                    try:
                        await release.wait()
                    finally:
                        self.active -= 1

                    title = json.loads(next(message.content for message in messages if isinstance(message, HumanMessage)))["paper"]["title"]
                    if title == "Paper 2":
                        raise RuntimeError("model unavailable")
                    return PaperRecommendationResult(
                        recommendation_stars=(
                            2 if title == "Paper 1" else 4
                        ),
                        recommendation_reason="相关。",
                    )

            model = Model()
            task = asyncio.create_task(
                RecommendationNode(artifact_store=store, model=as_chat_model(model))(
                    {
                        "child_run_id": child_run_id,
                        "abstract_manifest_artifact_ref": artifact.artifact_uri,
                        "query_understanding": (
                            _RECOMMENDATION_QUERY_UNDERSTANDING
                        ),
                    }
                )
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            self.assertEqual(model.max_active, 3)
            release.set()
            update = await task

            manifest = await store.read_json_uri(
                update["recommendation_manifest_artifact_ref"]
            )
            self.assertEqual(
                [
                    paper["paper_info"]["title"]
                    for paper in manifest["papers"]
                ],
                ["Paper 0", "Paper 3"],
            )
            self.assertEqual(
                manifest["removed_low_recommendation"][0]["title"],
                "Paper 1",
            )
            self.assertEqual(
                manifest["removed_recommendation_failed"][0]["title"],
                "Paper 2",
            )
            self.assertEqual(update["progress"]["recommendation_failed"], 1)

    async def test_recommendation_cancellation_cancels_active_scores(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            child_run_id = "child-cancel-recommendation"
            artifact = await store.write_json(
                run_id=child_run_id,
                step_key="abstract_enrichment",
                source="test",
                kind="paper_info_abstract_manifest_json",
                payload={
                    "papers": [
                        _concurrent_paper(index)
                        for index in range(3)
                    ]
                },
            )
            started = asyncio.Event()

            class Model:
                def __init__(self) -> None:
                    self.active = 0
                    self.cancelled = 0

                async def ainvoke(self, _messages: list):
                    self.active += 1
                    if self.active == 3:
                        started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        self.cancelled += 1
                        raise
                    finally:
                        self.active -= 1

            model = Model()
            task = asyncio.create_task(
                RecommendationNode(artifact_store=store, model=as_chat_model(model))(
                    {
                        "child_run_id": child_run_id,
                        "abstract_manifest_artifact_ref": artifact.artifact_uri,
                        "query_understanding": (
                            _RECOMMENDATION_QUERY_UNDERSTANDING
                        ),
                    }
                )
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

            self.assertEqual(model.cancelled, 3)

    async def test_crossref_limits_concurrency_and_preserves_results(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            child_run_id = "child-concurrent-crossref"
            artifact = await store.write_json(
                run_id=child_run_id,
                step_key="enrich_papers",
                source="test",
                kind="paper_info_enrichment_manifest_json",
                payload={
                    "papers": [
                        _concurrent_paper(0, doi="10.1/paper-0"),
                        *[
                            _concurrent_paper(index)
                            for index in range(1, 5)
                        ],
                    ]
                },
            )
            started = asyncio.Event()
            release = asyncio.Event()
            calls: list[dict] = []
            active = 0
            max_active = 0

            async def crossref_search(params: dict, _context: object) -> dict:
                nonlocal active, max_active
                calls.append(params)
                active += 1
                max_active = max(max_active, active)
                if active == 4:
                    started.set()
                try:
                    await release.wait()
                finally:
                    active -= 1

                if "doi" in params:
                    return {"ok": True, "papers": []}
                if params["title"] == "Paper 1":
                    return {
                        "ok": True,
                        "papers": [{"title": "Different Paper"}],
                    }
                if params["title"] == "Paper 2":
                    return {"ok": False, "error": "service unavailable"}
                return {
                    "ok": True,
                    "papers": [
                        {
                            "title": params["title"],
                            "authors": "Ada Lovelace",
                            "citation_count": 1,
                        }
                    ],
                }

            task = asyncio.create_task(
                CrossrefMetadataEnrichmentNode(
                    artifact_store=store,
                    crossref_search=crossref_search,
                )(
                    {
                        "child_run_id": child_run_id,
                        "enrichment_manifest_artifact_ref": (
                            artifact.artifact_uri
                        ),
                    }
                )
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            self.assertEqual(max_active, 4)
            release.set()
            update = await task

            manifest = await store.read_json_uri(
                update["crossref_enrichment_manifest_artifact_ref"]
            )
            self.assertEqual(
                [
                    paper["paper_info"]["title"]
                    for paper in manifest["papers"]
                ],
                [f"Paper {index}" for index in range(5)],
            )
            self.assertEqual(
                [
                    paper["crossref_metadata"]["status"]
                    for paper in manifest["papers"]
                ],
                ["matched", "not_matched", "failed", "matched", "matched"],
            )
            self.assertEqual(update["progress"]["crossref_matched"], 3)
            self.assertEqual(update["progress"]["crossref_unmatched"], 1)
            self.assertEqual(update["progress"]["crossref_failed"], 1)
            self.assertIn({"doi": "10.1/paper-0"}, calls)
            self.assertIn(
                {
                    "title": "Paper 0",
                    "limit": 5,
                    "author": "Ada Lovelace",
                },
                calls,
            )

    async def test_crossref_cancellation_cancels_active_queries(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            child_run_id = "child-cancel-crossref"
            artifact = await store.write_json(
                run_id=child_run_id,
                step_key="enrich_papers",
                source="test",
                kind="paper_info_enrichment_manifest_json",
                payload={
                    "papers": [
                        _concurrent_paper(index)
                        for index in range(4)
                    ]
                },
            )
            started = asyncio.Event()
            active = 0
            cancelled = 0

            async def crossref_search(_params: dict, _context: object) -> dict:
                nonlocal active, cancelled
                active += 1
                if active == 4:
                    started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled += 1
                    raise
                finally:
                    active -= 1

            task = asyncio.create_task(
                CrossrefMetadataEnrichmentNode(
                    artifact_store=store,
                    crossref_search=crossref_search,
                )(
                    {
                        "child_run_id": child_run_id,
                        "enrichment_manifest_artifact_ref": (
                            artifact.artifact_uri
                        ),
                    }
                )
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

            self.assertEqual(cancelled, 4)

    async def test_crossref_keeps_doi_fallback_in_the_same_slot(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            child_run_id = "child-crossref-fallback-slot"
            artifact = await store.write_json(
                run_id=child_run_id,
                step_key="enrich_papers",
                source="test",
                kind="paper_info_enrichment_manifest_json",
                payload={
                    "papers": [
                        _concurrent_paper(0, doi="10.1/paper-0"),
                        *[
                            _concurrent_paper(index)
                            for index in range(1, 5)
                        ],
                    ]
                },
            )
            initial_started = asyncio.Event()
            doi_release = asyncio.Event()
            fallback_started = asyncio.Event()
            fallback_release = asyncio.Event()
            fifth_started = asyncio.Event()
            fifth_release = asyncio.Event()
            other_release = asyncio.Event()
            initial_calls = 0

            async def crossref_search(params: dict, _context: object) -> dict:
                nonlocal initial_calls
                if "doi" in params:
                    initial_calls += 1
                    await doi_release.wait()
                    return {"ok": True, "papers": []}

                title = params["title"]
                if title == "Paper 0":
                    fallback_started.set()
                    await fallback_release.wait()
                elif title == "Paper 4":
                    fifth_started.set()
                    await fifth_release.wait()
                else:
                    initial_calls += 1
                    if initial_calls == 4:
                        initial_started.set()
                    await other_release.wait()

                return {
                    "ok": True,
                    "papers": [
                        {"title": title, "authors": "Ada Lovelace"}
                    ],
                }

            task = asyncio.create_task(
                CrossrefMetadataEnrichmentNode(
                    artifact_store=store,
                    crossref_search=crossref_search,
                )(
                    {
                        "child_run_id": child_run_id,
                        "enrichment_manifest_artifact_ref": (
                            artifact.artifact_uri
                        ),
                    }
                )
            )
            await asyncio.wait_for(initial_started.wait(), timeout=1)
            doi_release.set()
            await asyncio.wait_for(fallback_started.wait(), timeout=1)
            self.assertFalse(fifth_started.is_set())
            fallback_release.set()
            await asyncio.wait_for(fifth_started.wait(), timeout=1)
            other_release.set()
            fifth_release.set()
            await task

    async def test_empty_recommendation_and_crossref_skip_external_calls(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            child_run_id = "child-empty-concurrent-nodes"
            crossref_artifact = await store.write_json(
                run_id=child_run_id,
                step_key="enrich_papers",
                source="test",
                kind="paper_info_enrichment_manifest_json",
                payload={"papers": []},
            )

            async def crossref_search(_params: dict, _context: object) -> dict:
                raise AssertionError("Crossref should not run for an empty manifest")

            crossref_update = await CrossrefMetadataEnrichmentNode(
                artifact_store=store,
                crossref_search=crossref_search,
            )(
                {
                    "child_run_id": child_run_id,
                    "enrichment_manifest_artifact_ref": (
                        crossref_artifact.artifact_uri
                    ),
                }
            )
            self.assertEqual(crossref_update["stage"], "resolving_venues")

            recommendation_artifact = await store.write_json(
                run_id=child_run_id,
                step_key="abstract_enrichment",
                source="test",
                kind="paper_info_abstract_manifest_json",
                payload={"papers": []},
            )

            class Model:
                async def ainvoke(self, _messages: list):
                    raise AssertionError(
                        "Recommendation should not run for an empty manifest"
                    )

            recommendation_update = await RecommendationNode(
                artifact_store=store,
                model=as_chat_model(Model()),
            )(
                {
                    "child_run_id": child_run_id,
                    "abstract_manifest_artifact_ref": (
                        recommendation_artifact.artifact_uri
                    ),
                    "query_understanding": _RECOMMENDATION_QUERY_UNDERSTANDING,
                }
            )
            self.assertEqual(recommendation_update["stage"], "failed")

    async def test_existing_task_is_bound_without_creating_a_new_task(self) -> None:
        with TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            child_run_id = "child-existing-task"
            recommendation = await store.write_json(
                run_id=child_run_id,
                step_key="recommend_papers",
                source="recommendation",
                kind="paper_recommendation_manifest_json",
                payload={
                    "papers": [
                        {"task_paper_relation_draft": {"search_task_id": None}}
                    ]
                },
            )
            client = Mock(add_query_task=AsyncMock())
            update = await CreatePaperSearchTaskNode(
                client=client,
                artifact_store=store,
            )(
                {
                    "child_run_id": child_run_id,
                    "paper_service_task_id": 24,
                    "recommendation_manifest_artifact_ref": recommendation.artifact_uri,
                }
            )

            client.add_query_task.assert_not_awaited()
            self.assertEqual(update["stage"], "persisting_papers")
            bound_manifest = await store.read_json_uri(
                update["task_bound_recommendation_manifest_artifact_ref"]
            )
            self.assertEqual(bound_manifest["search_task_id"], 24)
            self.assertEqual(
                bound_manifest["papers"][0]["task_paper_relation_draft"]
                ["search_task_id"],
                24,
            )

    async def test_venue_resolution_uses_raw_dict_handler(self) -> None:
        with TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            calls: list[dict] = []

            async def fake_venue(params: dict, _context: object) -> dict:
                calls.append(params)
                return {
                    "ok": True,
                    "data": {
                        "publication_name": "Journal of Testing",
                        "abbreviation": "J Test",
                        "official_rank": {},
                        "custom_rank": [],
                    },
                }

            with patch(
                "app.llm.graph.workflows.paper_search.nodes.venue."
                "easy_scholar_venue_handler",
                new=fake_venue,
            ):
                node = VenueResolutionNode(
                    artifact_store=store,
                    cache_store=VenueCacheStore(
                        Path(directory) / "venue_cache.json"
                    ),
                    model=as_chat_model(object()),
                )
                venue, source = await node.resolve_venue(
                    title="A Paper",
                    venue_candidates=["Journal of Testing"],
                )

            self.assertEqual(
                calls,
                [{"publication_name": "Journal of Testing"}],
            )
            self.assertEqual(source, "easy_scholar")
            self.assertIsNotNone(venue)
            self.assertEqual(venue.standard_name, "Journal of Testing")


class PaperSearchSingleSourceValidationTests(unittest.TestCase):
    def test_workflow_has_search_and_review_nodes(self) -> None:
        nodes = build_paper_search_workflow(model=as_chat_model(object())).get_graph().nodes
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
        graph = build_paper_search_workflow(model=as_chat_model(object())).get_graph()
        targets = {
            edge.target
            for edge in graph.edges
            if edge.source == "plan_search"
        }

        self.assertIn("search", targets)
        self.assertNotIn("confirm", targets)

    def test_existing_task_routes_through_status_update_before_planning(self) -> None:
        graph = build_paper_search_workflow(model=as_chat_model(object())).get_graph()
        initialize_targets = {
            edge.target for edge in graph.edges if edge.source == "initialize"
        }
        status_targets = {
            edge.target
            for edge in graph.edges
            if edge.source == "update_task_status"
        }

        self.assertIn("plan_search", initialize_targets)
        self.assertIn("update_task_status", initialize_targets)
        self.assertIn("plan_search", status_targets)
        self.assertIn("finalize_result", status_targets)

    def test_review_routes_to_search_or_enrichment(self) -> None:
        graph = build_paper_search_workflow(model=as_chat_model(object())).get_graph()
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
        schema = PaperSearchParams.model_json_schema()
        self.assertNotIn("constraints", schema["properties"])
        self.assertIn("task_id", schema["properties"])

    def test_delegation_accepts_only_positive_integer_task_ids(self) -> None:
        self.assertEqual(
            PaperSearchParams.model_validate(
                {"prompt": "检索 RAG 方法论文", "task_id": 24}
            ).task_id,
            24,
        )
        for value in (True, 0, -1, "24"):
            with self.assertRaises(ValidationError):
                PaperSearchParams.model_validate(
                    {"prompt": "检索 RAG 方法论文", "task_id": value}
                )

    def test_state_does_not_include_unused_pdf_download_flag(self) -> None:
        self.assertNotIn("download_pdfs", PaperSearchWorkflowState.__annotations__)

    def test_subgraph_only_uses_google_scholar(self) -> None:
        names = {schema["name"] for schema in build_tool_registry().tool_schemas()}
        self.assertIn("google_scholar_search", names)
        self.assertIn("arxiv_search", names)
        self.assertIn("dblp_search", names)
        self.assertEqual(set(SEARCH_HANDLERS), {"Google Scholar"})


SEARCH_MODULE = "app.llm.tools.search_tools.google_scholar.search_google_scholar"


class SearchMetadataTests(unittest.TestCase):
    def test_crossref_zero_year_uses_next_valid_publication_date(self):
        self.assertEqual(_published_date({
            "published-print": {"date-parts": [[0]]},
            "issued": {"date-parts": [[2023, 5, 2]]},
        }), "2023-05-02")
        self.assertIsNone(_published_date({"issued": {"date-parts": [[0]]}}))

    def test_scholar_year_and_snippet_survive_normalization(self):
        paper = _parse_google_scholar_item({
            "title": "A paper",
            "publication_info": {"summary": "A Author - Journal, 2023 - publisher.org"},
            "snippet": "A source abstract fragment …",
        })
        info = _to_paper_info_draft(paper, "Google Scholar")
        self.assertEqual(paper["year"], 2023)
        self.assertEqual(info["publish_date"], "2023-01-01")
        self.assertEqual(info["publish_date_precision"], "year")
        self.assertEqual(info["paper_abstract"], paper["snippet"])
        self.assertEqual(info["abstract_source"], "search_snippet")

    def test_experiment_year_in_snippet_is_not_publication_year(self):
        paper = _parse_google_scholar_item({
            "title": "A paper", "snippet": "Evaluated on the 2020 dataset",
        })
        self.assertIsNone(paper["year"])
        self.assertIsNone(_to_paper_info_draft(paper, "Google Scholar")["publish_date"])

    def test_zero_and_missing_year_remain_unknown(self):
        for year in (0, "0", None):
            with self.subTest(year=year):
                info = _to_paper_info_draft({"title": "A paper", "year": year}, "Google Scholar")
                self.assertIsNone(info["publish_date"])

    def test_full_date_wins_over_year(self):
        info = _to_paper_info_draft({
            "title": "A paper", "published": "2024-03-02", "year": 2023,
        }, "Google Scholar")
        self.assertEqual(info["publish_date"], "2024-03-02")

    def test_crossref_replaces_year_precision_and_snippet(self):
        paper = {"paper_info": _to_paper_info_draft({
            "title": "A paper", "year": 2023, "abstract": "Fragment",
            "abstract_source": "search_snippet",
        }, "Google Scholar")}
        CrossrefMetadataEnrichmentNode()._merge(paper, {
            "published": "2023-07-12", "abstract": "Complete abstract",
        }, "title")
        self.assertEqual(paper["paper_info"]["publish_date"], "2023-07-12")
        self.assertEqual(paper["paper_info"]["paper_abstract"], "Complete abstract")
        self.assertEqual(paper["paper_info"]["abstract_source"], "crossref")

    def test_dedup_keeps_complete_abstract_over_longer_snippet(self):
        current = {"paper_info": _to_paper_info_draft({
            "title": "A paper", "abstract": "Full abstract", "published": "2024-03-02",
        }, "Crossref")}
        incoming = {"paper_info": _to_paper_info_draft({
            "title": "A paper", "abstract": "Much longer search fragment …",
            "abstract_source": "search_snippet", "year": 2024,
        }, "Google Scholar")}
        merged = _merge_envelope(current, incoming)
        self.assertEqual(merged["paper_info"]["paper_abstract"], "Full abstract")
        self.assertEqual(merged["paper_info"]["publish_date"], "2024-03-02")


class SearchMetadataFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_filter_and_persist_keep_year_and_abstract(self):
        with TemporaryDirectory() as directory:
            store = LocalArtifactStore(str(Path(directory) / "artifacts"))
            cache = PaperCacheStore(Path(directory) / "cache.json")
            raw_item = {
                "title": "A paper", "result_id": "result-1",
                "publication_info": {
                    "summary": "A Author - Journal, 2023 - publisher.org",
                    "authors": [{"name": "A Author"}],
                },
                "snippet": "A source abstract fragment …",
            }
            with patch(f"{SEARCH_MODULE}._build_serpapi_params", return_value={}), patch(
                f"{SEARCH_MODULE}._request_serpapi",
                new=AsyncMock(return_value={"organic_results": [raw_item, raw_item]}),
            ):
                search = await google_scholar_search_service({"query": "test"}, None)
            self.assertEqual(search["returned_count"], 1)
            raw = await store.write_json(
                run_id="metadata-test", step_key="search", source="Google Scholar",
                kind="raw", payload=search,
            )
            filtered = await NormalizeDeduplicateFilterNode(
                artifact_store=store, paper_cache_store=cache,
                paper_lookup=AsyncMock(return_value={"ok": True, "result": {"matches": []}}),
            )({
                "child_run_id": "metadata-test", "raw_result_artifact_refs": [raw.artifact_uri],
                "query_understanding": {"topic": "test", "intent": "method", "reasoning": "test"},
                "search_tag": {},
            })
            self.assertEqual(filtered["stage"], "reviewing_search", filtered)
            manifest = await store.read_json_uri(filtered["normalized_manifest_artifact_ref"])
            paper = manifest["papers"][0]
            paper["task_paper_relation_draft"] = {
                "search_task_id": 7, "paper_id": None,
                "recommendation_stars": 4, "recommendation_reason": "Matches topic",
            }
            recommended = await store.write_json(
                run_id="metadata-test", step_key="recommend", source="test",
                kind="recommended", payload={"papers": [paper]},
            )
            captured = []

            async def save_papers(items):
                captured.extend(items)
                return {"ok": True, "result": {"papers": [
                    {"client_key": items[0]["client_key"], "paper_id": 11},
                ]}}

            persisted = await PersistRecommendedPapersNode(
                artifact_store=store, paper_cache_store=cache,
                save_papers=save_papers,
                save_task_papers=AsyncMock(return_value={"ok": True, "result": {"saved_count": 1}}),
            )({
                "child_run_id": "metadata-test", "paper_service_task_id": 7,
                "recommendation_manifest_artifact_ref": recommended.artifact_uri,
            })
            self.assertEqual(persisted["stage"], "completed", persisted)
            request = PaperServiceGrpcClient._build_request(captured[0])
            self.assertTrue(request.HasField("published_date"))
            self.assertEqual(request.published_date.ToDatetime().year, 2023)
            self.assertEqual(request.paper_abstract, raw_item["snippet"])

    async def test_later_page_failure_retains_results(self):
        first_page = {
            "organic_results": [{"title": "A paper", "result_id": "1"}],
            "serpapi_pagination": {"next": "next-page"},
        }
        with patch(f"{SEARCH_MODULE}._build_serpapi_params", return_value={}), patch(
            f"{SEARCH_MODULE}._request_serpapi",
            new=AsyncMock(side_effect=[first_page, SerpApiSearchError("page failed")]),
        ):
            result = await google_scholar_search_service(
                {"query": "test", "num": 1, "total_limit": 2, "max_pages": 2}, None,
            )
        self.assertEqual(result["returned_count"], 1)
        self.assertEqual(result["status"], "partial_failed")
        self.assertFalse(result["metadata"]["pagination"]["has_next"])

    async def test_first_page_failure_is_not_an_empty_success(self):
        with patch(f"{SEARCH_MODULE}._build_serpapi_params", return_value={}), patch(
            f"{SEARCH_MODULE}._request_serpapi",
            new=AsyncMock(side_effect=SerpApiSearchError("page failed")),
        ):
            with self.assertRaisesRegex(SerpApiSearchError, "page failed"):
                await google_scholar_search_service({"query": "test"}, None)

    async def test_http_error_does_not_expose_request_api_key(self):
        request = httpx.Request("GET", "https://example.org/search?api_key=test-secret")
        response = httpx.Response(403, request=request, text="Forbidden")
        error = httpx.HTTPStatusError("HTTP failure at " + str(request.url), request=request, response=response)
        with patch(f"{SEARCH_MODULE}._build_serpapi_params", return_value={}), patch(
            f"{SEARCH_MODULE}._request_serpapi", new=AsyncMock(side_effect=error),
        ):
            with self.assertRaises(SerpApiSearchError) as caught:
                await google_scholar_search_service({"query": "test"}, None)
        self.assertEqual(str(caught.exception), "Forbidden")

    async def test_filter_deduplicates_and_preserves_date_precision(self):
        from app.llm.graph.workflows.paper_search_schemas import PromptUnderstandingArgs

        papers, conflicts, rejected, filtered = NormalizeDeduplicateFilterNode._normalize_payloads([
            {"source": "google_scholar", "papers": [
                {"title": "A paper", "authors": ["A Author"], "year": 2023,
                 "abstract": "Snippet …", "abstract_source": "search_snippet"},
                {"title": "A paper", "authors": ["A Author"], "published": "2023-05-02",
                 "abstract": "Full abstract"},
                {"title": "Old paper", "authors": ["A Author"], "year": 2010},
            ]},
        ], PromptUnderstandingArgs(
            topic="test", intent="method", reasoning="test", yearFrom=2020, yearTo=2025,
        ))
        self.assertEqual(len(papers), 1)
        self.assertEqual((conflicts, rejected, filtered), ([], [], 1))
        self.assertEqual(papers[0]["paper_info"]["publish_date"], "2023-05-02")
        self.assertEqual(papers[0]["paper_info"]["paper_abstract"], "Full abstract")

    async def test_new_task_creation_binds_recommendation_and_marks_running(self):
        with TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            artifact = await store.write_json(
                run_id="create-task-test", step_key="recommend", source="test", kind="recommended",
                payload={"papers": [{"task_paper_relation_draft": {"search_task_id": None}}]},
            )
            client = AsyncMock()
            client.add_query_task.return_value = {"ok": True, "result": {"task_id": 42}}
            client.update_task_status.return_value = {"ok": True}
            update = await CreatePaperSearchTaskNode(client=client, artifact_store=store)({
                "child_run_id": "create-task-test", "original_prompt": "Find papers",
                "query_understanding": {"topic": "test", "intent": "method", "reasoning": "test"},
                "search_tag": {}, "recommendation_manifest_artifact_ref": artifact.artifact_uri,
                "progress": {"recommendation_kept": 1},
            })
            self.assertEqual(update["stage"], "persisting_papers", update)
            self.assertEqual(update["paper_service_task_id"], 42)
            self.assertEqual(update["remote_task_state"], "SEARCH_RUNNING")
            self.assertEqual(update["progress"]["recommendation_kept"], 1)
            bound = await store.read_json_uri(update["task_bound_recommendation_manifest_artifact_ref"])
            self.assertEqual(bound["papers"][0]["task_paper_relation_draft"]["search_task_id"], 42)
            client.add_query_task.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
