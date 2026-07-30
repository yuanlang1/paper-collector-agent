import json
import tempfile
import unittest
from pathlib import Path

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.paper_search.nodes.filter import (
    NormalizeDeduplicateFilterNode,
)
from app.llm.graph.workflows.paper_search.nodes.paper_cache import (
    PaperCacheStore,
)
from app.llm.tools.task_tools.search_task.args import (
    PromptUnderstandingArgs,
    SearchTagArgs,
)


class FilterPaperCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_hit_becomes_existing_paper_without_lookup(self) -> None:
        async def paper_lookup(_candidates):
            self.fail("cache hit should not call paper lookup")

        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            paper_cache = PaperCacheStore(Path(directory) / "paper_cache.json")
            await paper_cache.upsert_saved_paper(
                {
                    "paper_id": 101,
                    "paper_info": {
                        "title": "A Cached Paper!",
                        "authors": "Ada Lovelace",
                        "doi": "10.1000/cached",
                        "paper_abstract": "cached abstract",
                        "ai_abstract": "缓存摘要",
                        "citations": 3,
                    },
                    "source_refs": [],
                    "venue_candidates": [],
                    "pdf_candidates": [],
                    "venue_resolution": {
                        "venue_id": 10,
                        "venue_draft": {"standard_name": "Test Journal"},
                    },
                }
            )
            raw = await store.write_json(
                run_id="run-1",
                step_key="search",
                source="Crossref",
                kind="test",
                payload={
                    "source": "Crossref",
                    "papers": [
                        {
                            "title": "A Cached Paper",
                            "authors": ["Ada Lovelace"],
                            "doi": "https://doi.org/10.1000/cached",
                            "abstract": "fresh source abstract",
                        }
                    ],
                },
            )
            node = NormalizeDeduplicateFilterNode(
                artifact_store=store,
                paper_lookup=paper_lookup,
                paper_cache_store=paper_cache,
            )
            result = await node(
                {
                    "run_id": "run-1",
                    "raw_result_artifact_refs": [raw.artifact_uri],
                    "query_understanding": PromptUnderstandingArgs(
                        topic="cache test",
                        intent="method",
                        reasoning="test",
                    ).model_dump(mode="json"),
                    "search_tag": SearchTagArgs().model_dump(mode="json"),
                }
            )
            self.assertEqual(result.get("stage"), "reviewing_search", result)
            existing_path = store.base_dir / result[
                "existing_papers_manifest_artifact_ref"
            ].removeprefix("artifact://")
            with existing_path.open(encoding="utf-8") as file:
                existing = json.load(file)["papers"]

        self.assertEqual(result["progress"]["existing_in_paper_cache"], 1)
        self.assertEqual(existing[0]["paper_id"], 101)
        self.assertEqual(existing[0]["paper_info"]["ai_abstract"], "缓存摘要")
        self.assertEqual(existing[0]["venue_resolution"]["venue_id"], 10)


if __name__ == "__main__":
    unittest.main()
