import tempfile
import unittest
from pathlib import Path

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.paper_search.nodes.paper_cache import (
    PaperCacheStore,
)
from app.llm.graph.workflows.paper_search.nodes.persist import (
    PersistRecommendedPapersNode,
)
from app.llm.graph.workflows.paper_search.nodes.venue import VenueCacheStore


class PersistRecommendedPapersNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_upstream_degradation_after_persistence(self) -> None:
        async def save_venues(_venues):
            self.fail("a missing venue must not be saved")

        async def save_papers(papers):
            self.assertEqual(papers[0]["paper_info"]["venue_id"], 0)
            return {
                "ok": True,
                "result": {
                    "papers": [
                        {
                            "client_key": papers[0]["client_key"],
                            "paper_id": 202,
                        }
                    ]
                },
            }

        async def save_relations(relations):
            self.assertEqual(relations[0]["paper_id"], 202)
            return {"ok": True, "result": {"saved_count": 1}}

        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            source = await store.write_json(
                run_id="run-without-venue",
                step_key="recommend_papers",
                source="recommendation",
                kind="test",
                payload={
                    "papers": [
                        {
                            "paper_id": None,
                            "paper_info": {
                                "title": "A Preprint",
                                "authors": "Ada Lovelace",
                                "publish_date": "2025-01-02",
                                "doi": "10.1000/preprint",
                            },
                            "venue_resolution": {
                                "status": "missing",
                                "venue_id": None,
                            },
                            "task_paper_relation_draft": {
                                "search_task_id": 1,
                                "paper_id": None,
                                "recommendation_stars": 4,
                                "recommendation_reason": "relevant",
                            },
                        }
                    ]
                },
            )
            node = PersistRecommendedPapersNode(
                artifact_store=store,
                save_venues=save_venues,
                save_papers=save_papers,
                save_task_papers=save_relations,
                venue_cache_store=VenueCacheStore(
                    Path(directory) / "venue_cache.json"
                ),
                paper_cache_store=PaperCacheStore(
                    Path(directory) / "paper_cache.json"
                ),
            )
            result = await node(
                {
                    "run_id": "run-without-venue",
                    "paper_service_task_id": 1,
                    "recommendation_manifest_artifact_ref": source.artifact_uri,
                    "degraded": True,
                }
            )

        self.assertEqual(result["stage"], "partial_failed")
        self.assertTrue(result["degraded"])

    async def test_persists_venue_paper_cache_and_relation(self) -> None:
        calls: list[str] = []

        async def save_venues(venues):
            calls.append("venues")
            self.assertEqual(venues[0]["standard_name"], "Test Conference")
            return {
                "ok": True,
                "result": {
                    "venues": [
                        {"standard_name": "Test Conference", "id": 101}
                    ]
                },
            }

        async def save_papers(papers):
            calls.append("papers")
            self.assertEqual(papers[0]["paper_info"]["venue_id"], 101)
            self.assertEqual(papers[0]["paper_info"]["published_date"], "2025-01-02")
            return {
                "ok": True,
                "result": {
                    "papers": [
                        {
                            "client_key": papers[0]["client_key"],
                            "paper_id": 202,
                            "status": "saved",
                        }
                    ]
                },
            }

        async def save_relations(relations):
            calls.append("relations")
            self.assertEqual(relations[0]["paper_id"], 202)
            return {"ok": True, "result": {"saved_count": len(relations)}}

        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            source = await store.write_json(
                run_id="run-1",
                step_key="recommend_papers",
                source="recommendation",
                kind="test",
                payload={
                    "papers": [
                        {
                            "paper_id": None,
                            "paper_info": {
                                "title": "A Paper",
                                "authors": "Ada Lovelace",
                                "publish_date": "2025-01-02",
                                "doi": "10.1000/example",
                                "citations": 3,
                            },
                            "pdf_download": {
                                "local_pdf_path": "C:/papers/example.pdf",
                                "artifact_uri": "artifact://papers/example.pdf",
                                "final_url": "https://example.test/example.pdf",
                            },
                            "venue_resolution": {
                                "status": "resolved",
                                "venue_id": None,
                                "venue_draft": {
                                    "standard_name": "Test Conference",
                                    "acronym": "TC",
                                    "type": 2,
                                },
                            },
                            "task_paper_relation_draft": {
                                "search_task_id": 1,
                                "paper_id": None,
                                "recommendation_stars": 4,
                                "recommendation_reason": "relevant",
                            },
                        },
                        {
                            "paper_id": 303,
                            "paper_info": {
                                "title": "An Existing Paper",
                                "authors": "Grace Hopper",
                                "publish_date": "2024-01-02",
                                "doi": "10.1000/existing",
                                "pdf_url": "https://example.test/existing.pdf",
                            },
                            "venue_resolution": {
                                "status": "missing",
                                "venue_id": 0,
                            },
                            "task_paper_relation_draft": {
                                "search_task_id": 1,
                                "paper_id": 303,
                                "recommendation_stars": 5,
                                "recommendation_reason": "existing and relevant",
                            },
                        }
                    ]
                },
            )
            venue_cache = VenueCacheStore(Path(directory) / "venue_cache.json")
            paper_cache = PaperCacheStore(Path(directory) / "paper_cache.json")
            node = PersistRecommendedPapersNode(
                artifact_store=store,
                save_venues=save_venues,
                save_papers=save_papers,
                save_task_papers=save_relations,
                venue_cache_store=venue_cache,
                paper_cache_store=paper_cache,
            )
            result = await node(
                {
                    "run_id": "run-1",
                    "paper_service_task_id": 1,
                    "recommendation_manifest_artifact_ref": source.artifact_uri,
                }
            )
            cached_papers, _ = paper_cache._read_sync()
            cached_venues = venue_cache._read_sync()

        self.assertEqual(result["stage"], "completed")
        self.assertEqual(calls, ["venues", "papers", "relations"])
        self.assertEqual(cached_papers["doi:10.1000/example"]["paper_id"], 202)
        self.assertEqual(
            cached_papers["doi:10.1000/example"]["pdf_download"],
            {"final_url": "https://example.test/example.pdf"},
        )
        self.assertEqual(
            cached_papers["doi:10.1000/existing"]["paper_id"],
            303,
        )
        self.assertEqual(
            cached_venues["test conference"]["venue_id"],
            101,
        )


if __name__ == "__main__":
    unittest.main()
