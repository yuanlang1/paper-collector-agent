"""Opt-in smoke test for the complete paper-search child workflow.

Run it explicitly with:
    $env:RUN_PAPER_SEARCH_LIVE_TEST = "1"
    .\\.venv310\\Scripts\\python.exe -m unittest \
        scripts.test_paper_seach_workflow.test_paper_search_live_integration -v

This test calls the configured LLM, search APIs, and paper-service gRPC APIs.
It creates a real search task and may persist venues, papers, and relations.
"""

import os
import uuid
import unittest

from app.config import settings
from app.infrastructure.grpc.grpc_channel_pool import paper_service_grpc_channel_pool
from app.infrastructure.nacos_registry import nacos_registry
from app.llm.graph.workflows.paper_search.workflow import (
    build_paper_search_workflow,
)


@unittest.skipUnless(
    os.getenv("RUN_PAPER_SEARCH_LIVE_TEST") == "1",
    "set RUN_PAPER_SEARCH_LIVE_TEST=1 to call real external services",
)
class PaperSearchLiveIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_with_real_llm_search_apis_and_grpc(self) -> None:
        original_limits = {
            "PAPER_SEARCH_TARGET_PAPER_COUNT": settings.PAPER_SEARCH_TARGET_PAPER_COUNT,
            "PAPER_SEARCH_ARXIV_MAX_PAGES": settings.PAPER_SEARCH_ARXIV_MAX_PAGES,
            "PAPER_SEARCH_ARXIV_PAGE_SIZE": settings.PAPER_SEARCH_ARXIV_PAGE_SIZE,
            "PAPER_SEARCH_ARXIV_TOTAL_LIMIT": settings.PAPER_SEARCH_ARXIV_TOTAL_LIMIT,
            "PAPER_SEARCH_DBLP_MAX_PAGES": settings.PAPER_SEARCH_DBLP_MAX_PAGES,
            "PAPER_SEARCH_DBLP_PAGE_SIZE": settings.PAPER_SEARCH_DBLP_PAGE_SIZE,
            "PAPER_SEARCH_DBLP_TOTAL_LIMIT": settings.PAPER_SEARCH_DBLP_TOTAL_LIMIT,
            "PAPER_SEARCH_GOOGLE_SCHOLAR_MAX_PAGES": settings.PAPER_SEARCH_GOOGLE_SCHOLAR_MAX_PAGES,
            "PAPER_SEARCH_GOOGLE_SCHOLAR_PAGE_SIZE": settings.PAPER_SEARCH_GOOGLE_SCHOLAR_PAGE_SIZE,
            "PAPER_SEARCH_GOOGLE_SCHOLAR_TOTAL_LIMIT": settings.PAPER_SEARCH_GOOGLE_SCHOLAR_TOTAL_LIMIT,
        }
        try:
            await nacos_registry.start()
            settings.PAPER_SEARCH_TARGET_PAPER_COUNT = 2
            for source in ("ARXIV", "DBLP", "GOOGLE_SCHOLAR"):
                setattr(settings, f"PAPER_SEARCH_{source}_MAX_PAGES", 1)
                setattr(settings, f"PAPER_SEARCH_{source}_PAGE_SIZE", 3)
                setattr(settings, f"PAPER_SEARCH_{source}_TOTAL_LIMIT", 3)

            result = await build_paper_search_workflow(
                skip_confirmation=True,
            ).ainvoke(
                {
                    "run_id": f"live-integration-{uuid.uuid4().hex}",
                    "original_prompt": (
                        "Find recent survey papers about retrieval-augmented "
                        "generation evaluation."
                    ),
                    "warnings": [],
                    "progress": {},
                }
            )
        finally:
            for name, value in original_limits.items():
                setattr(settings, name, value)
            await paper_service_grpc_channel_pool.close()
            await nacos_registry.stop()

        self.assertIsInstance(
            result.get("paper_service_task_id"),
            int,
            msg=str(result),
        )
        self.assertIn("source_search_stats", result)
        self.assertIn(
            result.get("stage"),
            {"completed", "partial_failed"},
            msg=result.get("error"),
        )
        self.assertIn("persisted_papers_manifest_artifact_ref", result)


if __name__ == "__main__":
    unittest.main()
