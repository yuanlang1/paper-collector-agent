import unittest
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.paper_search.nodes.build_search_tag import (
    BuildSearchTagNode,
)
from app.llm.graph.workflows.paper_search.nodes.confirm import (
    paper_search_confirm_node,
)
from app.llm.graph.workflows.paper_search.nodes.create_task import (
    CreatePaperSearchTaskNode,
)
from app.llm.graph.workflows.paper_search.nodes.persist import (
    PersistRecommendedPapersNode,
)
from app.llm.graph.workflows.paper_search.workflow import (
    _terminal_target,
    build_paper_search_workflow,
)
from app.llm.streaming.timeline import PAPER_SEARCH_TIMELINE, _step_for_node
from app.llm.streaming.visual_adapter import PAPER_SEARCH_NODE_PHASE


class DeferredTaskCreationTests(unittest.IsolatedAsyncioTestCase):
    async def test_created_task_binds_recommendations_before_persisting(self):
        class TaskClient:
            async def add_query_task(self, _payload):
                return {"ok": True, "result": {"task_id": 88}}

            async def update_task_status(self, **_kwargs):
                return {"ok": True}

        with TemporaryDirectory() as directory:
            artifact_store = LocalArtifactStore(directory)
            recommendation_artifact = await artifact_store.write_json(
                run_id="run-1",
                step_key="recommend_papers",
                source="recommendation",
                kind="paper_info_recommendation_manifest_json",
                payload={
                    "papers": [
                        {
                            "paper_id": 17,
                            "paper_info": {
                                "title": "AQA Paper",
                                "doi": "10.1000/aqa",
                            },
                            "task_paper_relation_draft": {
                                "search_task_id": None,
                                "paper_id": 17,
                                "recommendation_stars": 4,
                                "recommendation_reason": "高度相关",
                            },
                        }
                    ]
                },
            )
            state = {
                "run_id": "run-1",
                "original_prompt": "AQA related papers",
                "search_tag": {
                    "yearTag": 0,
                    "paperTag": [1, 2],
                    "sourceTag": ["arXiv", "DBLP", "Google Scholar"],
                },
                "query_understanding": {
                    "topic": "AQA",
                    "intent": "survey",
                    "reasoning": "Find relevant papers",
                },
                "recommendation_manifest_artifact_ref": (
                    recommendation_artifact.artifact_uri
                ),
            }

            created = await CreatePaperSearchTaskNode(
                client=TaskClient(),
                artifact_store=artifact_store,
            )(state)

            bound_artifact_uri = created[
                "task_bound_recommendation_manifest_artifact_ref"
            ]
            self.assertNotEqual(
                bound_artifact_uri,
                recommendation_artifact.artifact_uri,
            )
            self.assertIsNone(
                artifact_store.read_json(recommendation_artifact)["papers"][0][
                    "task_paper_relation_draft"
                ]["search_task_id"]
            )
            bound_manifest = await artifact_store.read_json_uri(
                bound_artifact_uri
            )
            self.assertEqual(
                bound_manifest["papers"][0]["task_paper_relation_draft"][
                    "search_task_id"
                ],
                88,
            )

            saved_relations = []

            async def save_task_papers(relations):
                saved_relations.extend(relations)
                return {"ok": True, "result": {"saved_count": len(relations)}}

            persist_result = await PersistRecommendedPapersNode(
                artifact_store=artifact_store,
                save_task_papers=save_task_papers,
            )(
                {
                    **state,
                    **created,
                }
            )

            self.assertEqual(persist_result["stage"], "completed")
            self.assertEqual(
                saved_relations,
                [
                    {
                        "task_id": 88,
                        "paper_id": 17,
                        "recommendation_stars": 4,
                        "recommendation_reason": "高度相关",
                    }
                ],
            )

    async def test_skipping_confirmation_starts_query_generation(self):
        node = BuildSearchTagNode(
            model=AsyncMock(),
            skip_confirmation=True,
        )
        node.model.ainvoke.return_value = {
            "yearTag": 0,
            "paperTag": [1, 2],
            "sourceTag": ["arXiv", "DBLP", "Google Scholar"],
        }

        result = await node(
            {
                "original_prompt": "AQA related papers",
                "query_understanding": {
                    "topic": "AQA",
                    "intent": "survey",
                    "reasoning": "Find relevant papers",
                },
                "paper_search_constraints": {},
            }
        )

        self.assertEqual(result["stage"], "generating_source_queries")

    async def test_approved_confirmation_starts_query_generation(self):
        state = {
            "original_prompt": "AQA related papers",
            "query_understanding": {
                "topic": "AQA",
                "intent": "survey",
                "reasoning": "Find relevant papers",
            },
            "search_tag": {
                "yearTag": 0,
                "paperTag": [1, 2],
                "sourceTag": ["arXiv", "DBLP", "Google Scholar"],
            },
        }
        with patch(
            "app.llm.graph.workflows.paper_search.nodes.confirm._interrupt_with_config",
            return_value={"decision": "approved"},
        ):
            result = await paper_search_confirm_node(state, {})

        self.assertEqual(result["stage"], "generating_source_queries")

    async def test_existing_task_continues_to_persist(self):
        result = await CreatePaperSearchTaskNode()(
            {"paper_service_task_id": 42}
        )

        self.assertEqual(result["stage"], "persisting_papers")
        self.assertEqual(result["status"], "running")

    async def test_existing_task_preserves_partial_failure(self):
        result = await CreatePaperSearchTaskNode()(
            {"paper_service_task_id": 42, "degraded": True}
        )

        self.assertEqual(result["stage"], "persisting_papers")
        self.assertEqual(result["status"], "partial_failed")

    def test_failed_deferred_task_creation_still_cleans_downloads(self):
        target = _terminal_target(
            {
                "stage": "failed",
                "status": "failed",
                "downloaded_pdf_paths": ["artifact-path/paper.pdf"],
            }
        )

        self.assertEqual(target, "cleanup_downloaded_pdfs")

    def test_task_creation_is_in_persist_and_sync_timeline_step(self):
        step = _step_for_node(PAPER_SEARCH_TIMELINE, "create_task")

        self.assertIsNotNone(step)
        self.assertEqual(step.key, "persist_and_sync")
        self.assertIn("create_task", step.starts_at)
        self.assertEqual(PAPER_SEARCH_NODE_PHASE["create_task"], "persist")

    def test_task_is_created_after_recommendation(self):
        graph = build_paper_search_workflow(skip_confirmation=True).get_graph()
        edges = {(edge.source, edge.target) for edge in graph.edges}

        self.assertIn(("recommend", "create_task"), edges)
        self.assertIn(("create_task", "persist"), edges)
        self.assertNotIn(("create_task", "generate_queries"), edges)


if __name__ == "__main__":
    unittest.main()
