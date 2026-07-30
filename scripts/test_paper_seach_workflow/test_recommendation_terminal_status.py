import tempfile
import unittest

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.paper_search.nodes.recommend import (
    PaperRecommendationResult,
    RecommendationNode,
)


class _FailingModel:
    async def ainvoke(self, _messages):
        raise RuntimeError("recommendation unavailable")


class _LowScoreModel:
    async def ainvoke(self, _messages):
        return PaperRecommendationResult(
            recommendation_stars=2,
            recommendation_reason="not relevant enough",
        )


class RecommendationTerminalStatusTests(
    unittest.IsolatedAsyncioTestCase
):
    async def _run(self, model) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            manifest = await store.write_json(
                run_id="recommend-run",
                step_key="abstract_enrichment",
                source="abstract",
                kind="test",
                payload={
                    "papers": [
                        {
                            "paper_info": {
                                "title": "Test Paper",
                                "authors": "Ada Lovelace",
                            }
                        }
                    ]
                },
            )
            return await RecommendationNode(
                artifact_store=store,
                model=model,
            )(
                {
                    "run_id": "recommend-run",
                    "paper_service_task_id": 1,
                    "abstract_manifest_artifact_ref": (
                        manifest.artifact_uri
                    ),
                    "query_understanding": {
                        "topic": "retrieval",
                        "intent": "method",
                        "reasoning": "test",
                    },
                    "warnings": [],
                    "progress": {},
                    "degraded": False,
                }
            )

    async def test_all_scoring_failures_are_failed(self) -> None:
        result = await self._run(_FailingModel())

        self.assertEqual(result["stage"], "failed")
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["degraded"])

    async def test_all_valid_low_scores_are_completed(self) -> None:
        result = await self._run(_LowScoreModel())

        self.assertEqual(result["stage"], "completed")
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["degraded"])


if __name__ == "__main__":
    unittest.main()
