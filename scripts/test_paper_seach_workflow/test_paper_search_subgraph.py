import unittest
from unittest.mock import patch

from app.infrastructure.task_service_grpc_client import TaskState
from app.llm.graph.workflows.paper_search.nodes.search import (
    SEARCH_HANDLERS,
)
from app.llm.graph.workflows.paper_search.workflow import (
    build_paper_search_workflow,
)


class PaperSearchSubgraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_compiled_graph_exposes_conditional_edges(self) -> None:
        graph = build_paper_search_workflow(skip_confirmation=True)
        edges = {
            (edge.source, edge.target)
            for edge in graph.get_graph(xray=True).edges
        }

        self.assertIn(
            ("supplemental_search", "search_arxiv"),
            edges,
        )
        self.assertIn(
            ("persist", "cleanup_downloaded_pdfs"),
            edges,
        )
        self.assertIn(
            ("cleanup_downloaded_pdfs", "update_task_status"),
            edges,
        )
        self.assertGreater(len(edges), 20)

    async def test_runs_complete_path_with_replaced_nodes(self) -> None:
        calls: list[str] = []

        def fake(name: str, result: dict):
            async def node(_state, *_args):
                calls.append(name)
                return result

            return node

        graph = build_paper_search_workflow(
            skip_confirmation=True,
            node_overrides={
                "intent_understanding": fake(
                    "intent",
                    {"stage": "building_search_tag"},
                ),
                "build_search_tag": fake(
                    "tag",
                    {"stage": "creating_task"},
                ),
                "create_task": fake(
                    "task",
                    {"stage": "generating_source_queries"},
                ),
                "generate_queries": fake("plan", {"stage": "searching"}),
                "search_arxiv": fake("arxiv", {}),
                "search_dblp": fake("dblp", {}),
                "search_google": fake("google", {}),
                "finalize_source_search": fake(
                    "finalize_search",
                    {"stage": "normalizing"},
                ),
                "filter": fake("filter", {"stage": "reviewing_search"}),
                "search_review": fake("review", {"stage": "enriching"}),
                "enrich": fake("enrich", {"stage": "enriching_crossref"}),
                "crossref_enrich": fake("crossref", {"stage": "resolving_venues"}),
                "venue": fake("venue", {"stage": "downloading_pdfs"}),
                "download_pdf": fake("pdf", {"stage": "enriching_abstract"}),
                "abstract_enrich": fake("abstract", {"stage": "recommending"}),
                "recommend": fake("recommend", {"stage": "persisting_papers"}),
                "persist": fake("persist", {"stage": "completed", "status": "completed"}),
                "cleanup_downloaded_pdfs": fake("cleanup", {}),
                "update_task_status": fake("task_status", {}),
            },
        )

        result = await graph.ainvoke(
            {"run_id": "run-test", "original_prompt": "test"}
        )

        self.assertEqual(result["stage"], "completed")
        self.assertEqual(
            result["paper_search_handoff"]["status"],
            "success",
        )
        self.assertEqual(
            calls,
            [
                "intent", "tag", "task", "plan", "arxiv", "dblp",
                "google", "finalize_search", "filter", "review", "enrich", "crossref",
                "venue", "pdf", "abstract", "recommend", "persist", "cleanup",
                "task_status",
            ],
        )

    async def test_search_exception_updates_remote_task_to_failed(
        self,
    ) -> None:
        class RecordingTaskClient:
            def __init__(self) -> None:
                self.calls = []

            async def update_task_status(self, **kwargs):
                self.calls.append(kwargs)
                return {
                    "ok": True,
                    "result": {"updated": True},
                    "error": None,
                }

        async def fake(_state, *_args):
            return {}

        async def initialize(_state):
            return {
                "stage": "intent_understanding",
                "status": "running",
            }

        async def intent(_state):
            return {
                "stage": "building_search_tag",
                "status": "running",
            }

        async def tag(_state):
            return {
                "stage": "creating_task",
                "status": "running",
            }

        async def create(_state):
            return {
                "stage": "generating_source_queries",
                "status": "running",
                "paper_service_task_id": 42,
            }

        async def plan(_state):
            return {
                "stage": "searching",
                "status": "running",
                "active_search_sources": ["arXiv"],
                "active_source_query_plans": [
                    {
                        "source": "arXiv",
                        "display_query": "RAG",
                        "reasoning": "test",
                        "arguments": {"max_results": 10},
                        "post_filters": {},
                    }
                ],
            }

        async def explode(_arguments, _db):
            raise RuntimeError("search transport crashed")

        cleanup_calls: list[dict] = []

        async def cleanup(state):
            cleanup_calls.append(state)
            return {}

        client = RecordingTaskClient()
        graph = build_paper_search_workflow(
            skip_confirmation=True,
            task_client=client,
            node_overrides={
                "initialize": initialize,
                "intent_understanding": intent,
                "build_search_tag": tag,
                "create_task": create,
                "generate_queries": plan,
                "search_dblp": fake,
                "search_google": fake,
                "cleanup_downloaded_pdfs": cleanup,
            },
        )

        with patch.dict(
            SEARCH_HANDLERS,
            {"arXiv": explode},
        ):
            result = await graph.ainvoke(
                {"run_id": "run-failed-search"},
                config={"configurable": {"db": None}},
            )

        self.assertEqual(result["stage"], "failed")
        self.assertEqual(
            result["paper_search_handoff"]["status"],
            "error",
        )
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(cleanup_calls), 1)
        self.assertEqual(
            client.calls[0]["task_state"],
            TaskState.SEARCH_FAILED,
        )


if __name__ == "__main__":
    unittest.main()
