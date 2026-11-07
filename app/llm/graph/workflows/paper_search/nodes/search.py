from langgraph.runtime import Runtime

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.context import PaperSearchContext
from app.llm.graph.workflows.paper_search.state import SearchWorkerState
from app.services.search_connection_executor import SearchConnectionError
from app.services.search_connection_executor import (
    search_executor as shared_search_executor,
)
from app.utils.search_adapters.contracts import failed_result


class SearchNode:
    def __init__(self, *, search_executor=None, artifact_store=None):
        self.executor = (
            search_executor if search_executor is not None else shared_search_executor
        )
        self.artifact_store = artifact_store or LocalArtifactStore()

    async def __call__(
        self, state: SearchWorkerState, runtime: Runtime[PaperSearchContext]
    ) -> dict:
        task = state["search_task"]
        provider = task["provider_id"]
        try:
            result = await self.executor.search(
                user_id=runtime.context["user_id"], task=task
            )
        except SearchConnectionError as exc:
            result = failed_result(provider, str(exc))
        artifact = await self.artifact_store.write_json(
            run_id=state["child_run_id"],
            step_key=f"search_{task['round']}_{provider}",
            source=provider,
            payload={
                "ok": result.status != "failed",
                "source": provider,
                "run_id": state["child_run_id"],
                "query": task["query_args"].get("query"),
                "papers": result.papers,
                "total_results": result.total_results,
                "metadata": {"provider_id": provider, "round": task["round"]},
            },
        )
        return {
            "search_results": {
                provider: {
                    "provider_id": provider,
                    "round": task["round"],
                    "status": result.status,
                    "artifact_ref": artifact.artifact_uri,
                    "returned_count": len(result.papers),
                    "pages_fetched": result.pages_fetched,
                    "raw_fetched_count": result.raw_fetched_count,
                    "next_cursor": result.next_cursor,
                    "warnings": result.warnings,
                    "error": result.error,
                    "stop_reason": result.stop_reason,
                }
            }
        }
