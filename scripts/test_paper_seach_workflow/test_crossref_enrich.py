import tempfile
import unittest
from pathlib import Path

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.paper_search.nodes.crossref_enrich import (
    CrossrefMetadataEnrichmentNode,
)
from app.llm.graph.workflows.paper_search.nodes.venue import (
    ModelVenueInfoDraft,
    VenueCacheStore,
    VenueResolutionNode,
)


class CrossrefEnrichmentNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_doi_match_enriches_metadata_and_type(self) -> None:
        async def search(params, _db):
            self.assertEqual(params, {"doi": "10.1000/example"})
            return {
                "ok": True,
                "papers": [
                    {
                        "doi": "10.1000/example",
                        "published": "2025-02-03",
                        "citation_count": 12,
                        "venue": "Test Conference",
                        "publication_type": "proceedings-article",
                        "landing_url": "https://doi.org/10.1000/example",
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            source = await store.write_json(
                run_id="run-1",
                step_key="paper_enrichment",
                source="enriched",
                kind="test",
                payload={
                    "papers": [
                        {
                            "paper_info": {
                                "title": "A Paper",
                                "authors": "Ada Lovelace",
                                "doi": "10.1000/example",
                                "publish_date": None,
                                "citations": 3,
                                "abstract_url": None,
                            },
                            "venue_candidates": [],
                        }
                    ]
                },
            )
            node = CrossrefMetadataEnrichmentNode(
                artifact_store=store,
                crossref_search=search,
            )

            result = await node(
                {
                    "run_id": "run-1",
                    "enrichment_manifest_artifact_ref": source.artifact_uri,
                }
            )
            output_path = store.base_dir / result[
                "crossref_enrichment_manifest_artifact_ref"
            ].removeprefix("artifact://")
            output = store.read_json(
                type("Artifact", (), {"path": str(output_path)})()
            )

        paper = output["papers"][0]
        self.assertEqual(result["stage"], "resolving_venues")
        self.assertEqual(paper["paper_info"]["publish_date"], "2025-02-03")
        self.assertEqual(paper["paper_info"]["citations"], 12)
        self.assertEqual(paper["venue_candidates"], ["Test Conference"])
        self.assertEqual(
            paper["crossref_metadata"]["paper_type_code"], 2
        )

    async def test_title_search_requires_exact_title_match(self) -> None:
        async def search(_params, _db):
            return {
                "ok": True,
                "papers": [
                    {
                        "title": "A Different Paper",
                        "authors": ["Ada Lovelace"],
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            source = await store.write_json(
                run_id="run-2",
                step_key="paper_enrichment",
                source="enriched",
                kind="test",
                payload={
                    "papers": [
                        {
                            "paper_info": {
                                "title": "A Paper",
                                "authors": "Ada Lovelace",
                                "doi": None,
                            }
                        }
                    ]
                },
            )
            node = CrossrefMetadataEnrichmentNode(
                artifact_store=store,
                crossref_search=search,
            )
            result = await node(
                {
                    "run_id": "run-2",
                    "enrichment_manifest_artifact_ref": source.artifact_uri,
                }
            )
            output_path = store.base_dir / result[
                "crossref_enrichment_manifest_artifact_ref"
            ].removeprefix("artifact://")
            output = store.read_json(
                type("Artifact", (), {"path": str(output_path)})()
            )

        self.assertEqual(
            output["papers"][0]["crossref_metadata"]["status"],
            "not_matched",
        )

    async def test_doi_not_found_falls_back_to_title_search(self) -> None:
        async def search(params, _db):
            if "doi" in params:
                return {
                    "ok": False,
                    "error": {"code": "CROSSREF_NOT_FOUND"},
                }
            return {
                "ok": True,
                "papers": [
                    {
                        "title": "A Paper",
                        "authors": ["Ada Lovelace"],
                        "doi": "10.1000/found-by-title",
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            source = await store.write_json(
                run_id="run-5",
                step_key="paper_enrichment",
                source="enriched",
                kind="test",
                payload={
                    "papers": [
                        {
                            "paper_info": {
                                "title": "A Paper",
                                "authors": "Ada Lovelace",
                                "doi": "10.1000/not-in-crossref",
                            }
                        }
                    ]
                },
            )
            node = CrossrefMetadataEnrichmentNode(
                artifact_store=store,
                crossref_search=search,
            )
            result = await node(
                {
                    "run_id": "run-5",
                    "enrichment_manifest_artifact_ref": source.artifact_uri,
                }
            )
            output_path = store.base_dir / result[
                "crossref_enrichment_manifest_artifact_ref"
            ].removeprefix("artifact://")
            output = store.read_json(
                type("Artifact", (), {"path": str(output_path)})()
            )

        self.assertEqual(result["progress"]["crossref_matched"], 1)
        self.assertEqual(
            output["papers"][0]["crossref_metadata"]["match_method"],
            "title",
        )

    async def test_search_failure_is_recorded_without_stopping_batch(self) -> None:
        async def search(_params, _db):
            return {
                "ok": False,
                "error": {"message": "rate limited"},
            }

        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            source = await store.write_json(
                run_id="run-4",
                step_key="paper_enrichment",
                source="enriched",
                kind="test",
                payload={
                    "papers": [
                        {
                            "paper_info": {
                                "title": "A Paper",
                                "doi": "10.1000/example",
                            }
                        }
                    ]
                },
            )
            node = CrossrefMetadataEnrichmentNode(
                artifact_store=store,
                crossref_search=search,
            )
            result = await node(
                {
                    "run_id": "run-4",
                    "enrichment_manifest_artifact_ref": source.artifact_uri,
                }
            )

        self.assertEqual(result["stage"], "resolving_venues")
        self.assertEqual(result["status"], "partial_failed")
        self.assertEqual(result["progress"]["crossref_failed"], 1)


class VenueResolutionNodeTests(unittest.IsolatedAsyncioTestCase):
    def test_model_empty_acronym_falls_back_to_standard_name(self) -> None:
        venue = ModelVenueInfoDraft.model_validate({
            "standard_name": "International Conference on Learning Representations",
            "acronym": "   ",
        })

        self.assertEqual(
            venue.acronym,
            "International Conference on Learning Representations",
        )

    async def test_crossref_type_overrides_cached_venue_type(self) -> None:
        async def easy_scholar(_params, _db):
            raise AssertionError("cache hit should skip EasyScholar")

        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            cache = VenueCacheStore(Path(directory) / "venue_cache.json")
            cache._write_sync({
                "test conference": {
                    "standard_name": "Test Conference",
                    "acronym": "TC",
                    "type": 0,
                }
            })
            source = await store.write_json(
                run_id="run-3",
                step_key="crossref_enrichment",
                source="crossref",
                kind="test",
                payload={
                    "papers": [
                        {
                            "paper_info": {"title": "A Paper"},
                            "venue_candidates": ["Test Conference"],
                            "crossref_metadata": {
                                "paper_type_code": 2,
                            },
                        }
                    ]
                },
            )
            node = VenueResolutionNode(
                artifact_store=store,
                cache_store=cache,
                model=object(),
                easy_scholar_handler=easy_scholar,
            )
            result = await node(
                {
                    "run_id": "run-3",
                    "crossref_enrichment_manifest_artifact_ref": (
                        source.artifact_uri
                    ),
                }
            )
            output_path = store.base_dir / result[
                "venue_manifest_artifact_ref"
            ].removeprefix("artifact://")
            output = store.read_json(
                type("Artifact", (), {"path": str(output_path)})()
            )

        self.assertEqual(
            output["papers"][0]["venue_resolution"]["venue_draft"]["type"],
            2,
        )

    async def test_paper_without_a_venue_is_retained_for_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            source = await store.write_json(
                run_id="run-arxiv",
                step_key="crossref_enrichment",
                source="crossref",
                kind="test",
                payload={
                    "papers": [
                        {
                            "paper_info": {
                                "title": "A Preprint",
                                "source": "arXiv",
                            },
                            "venue_candidates": [],
                        }
                    ]
                },
            )
            node = VenueResolutionNode(
                artifact_store=store,
            )
            result = await node(
                {
                    "run_id": "run-arxiv",
                    "crossref_enrichment_manifest_artifact_ref": (
                        source.artifact_uri
                    ),
                }
            )
            output_path = store.base_dir / result[
                "venue_manifest_artifact_ref"
            ].removeprefix("artifact://")
            output = store.read_json(
                type("Artifact", (), {"path": str(output_path)})()
            )

        self.assertEqual(len(output["papers"]), 1)
        self.assertEqual(
            output["papers"][0]["venue_resolution"]["status"],
            "missing",
        )
        self.assertEqual(result["progress"]["venue_missing"], 1)


if __name__ == "__main__":
    unittest.main()
