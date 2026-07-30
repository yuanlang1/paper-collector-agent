"""Run the formal paper-search child workflow from the command line.

Default mode uses real LLM and search APIs, while task creation and persistence
are replaced so the script does not write remote paper-service data.

Set USE_REAL_PAPER_SERVICE=true to start the FastAPI lifespan and use real
Nacos/gRPC for task, venue, paper, and task-paper relation persistence.
"""

import asyncio
import json
import os
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.paper_search.workflow import (
    build_paper_search_workflow,
)


class FakePaperServiceClient:
    async def add_query_task(self, request: dict) -> dict:
        return {
            "ok": True,
            "result": {"task_id": 10001, "request": request},
            "error": None,
        }

    async def update_task_status(self, **_kwargs) -> dict:
        return {"ok": True, "result": {"updated": True}, "error": None}


def build_fake_persist_node(artifact_store: LocalArtifactStore):
    async def fake_persist_node(state: dict) -> dict:
        run_id = str(state["run_id"])
        recommendation_ref = state.get("recommendation_manifest_artifact_ref")
        if not isinstance(recommendation_ref, str):
            raise ValueError("recommendation manifest is missing")

        path = artifact_store.base_dir / recommendation_ref.removeprefix("artifact://")
        recommendation = json.loads(path.read_text(encoding="utf-8"))
        papers = recommendation.get("papers") or []
        first_paper = papers[0] if papers else {}
        title = ((first_paper.get("paper_info") or {}).get("title") if isinstance(first_paper, dict) else "") or "dry-run paper"
        artifact = await artifact_store.write_json(
            run_id=run_id,
            step_key="persist_recommended_papers",
            source="test",
            kind="persisted_papers_manifest_json",
            payload={
                "run_id": run_id,
                "input_manifest": recommendation_ref,
                "results": [
                    {
                        "paper_id": 10001,
                        "title": title,
                        "task_paper_save_status": "saved",
                    }
                ] if papers else [],
            },
        )
        return {
            "stage": "completed",
            "status": "completed",
            "persisted_papers_manifest_artifact_ref": artifact.artifact_uri,
            "progress": {**state.get("progress", {}), "persistence_saved": int(bool(papers))},
        }

    return fake_persist_node


def _print_state(result: dict) -> None:
    keys = (
        "run_id",
        "query_understanding",
        "search_tag",
        "confirmation_decision",
        "paper_service_task_id",
        "source_query_plans",
        "source_search_stats",
        "supplemental_search_round",
        "normalized_manifest_artifact_ref",
        "crossref_enrichment_manifest_artifact_ref",
        "venue_manifest_artifact_ref",
        "recommendation_manifest_artifact_ref",
        "persisted_papers_manifest_artifact_ref",
        "progress",
        "warnings",
        "stage",
        "status",
        "error",
    )
    print(
        json.dumps(
            {key: result.get(key) for key in keys},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


async def run_graph(*, use_real_paper_service: bool) -> None:
    overrides = None
    task_client = None
    if not use_real_paper_service:
        task_client = FakePaperServiceClient()
        artifact_store = LocalArtifactStore()
        overrides = {
            "persist": build_fake_persist_node(artifact_store),
        }

    graph = build_paper_search_workflow(
        task_client=task_client,
        checkpointer=MemorySaver(),
        node_overrides=overrides,
    )
    config = {"configurable": {"thread_id": "paper-search-workflow-demo"}}
    initial_state = {
        "run_id": f"paper-search-workflow-test-{uuid.uuid4().hex}",
        "original_prompt": (
            "Find survey papers about retrieval-augmented generation "
            "evaluation, focusing on benchmarks and excluding healthcare."
        ),
        "warnings": [],
        "progress": {},
    }

    interrupted = await graph.ainvoke(initial_state, config=config)
    interrupt = interrupted.get("__interrupt__", ())
    if not interrupt:
        raise RuntimeError(f"workflow did not request confirmation: {interrupted}")

    print("Confirmation request:")
    print(json.dumps(interrupt[0].value, ensure_ascii=False, indent=2))

    result = await graph.ainvoke(
        Command(resume={"decision": "approved"}),
        config=config,
    )
    print("\nFinal workflow state:")
    _print_state(result)


async def main() -> None:
    use_real_paper_service = (
        os.getenv("USE_REAL_PAPER_SERVICE", "false").lower() == "true"
    )
    if not use_real_paper_service:
        print("Test mode: real LLM/search APIs; fake task and persistence gRPC")
        await run_graph(use_real_paper_service=False)
        return

    print("Integration mode: real LLM/search APIs and Nacos/paper-service gRPC")
    from app.main import app

    async with app.router.lifespan_context(app):
        await run_graph(use_real_paper_service=True)


if __name__ == "__main__":
    asyncio.run(main())
