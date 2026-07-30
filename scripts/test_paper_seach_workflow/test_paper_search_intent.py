import asyncio
import json
import os
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.llm.graph.workflows.paper_search.nodes.intent import (
    IntentUnderstandingNode,
)
from app.llm.graph.workflows.paper_search.nodes.build_search_tag import (
    BuildDefaultSearchTagNode,
)
from app.llm.graph.workflows.paper_search.nodes.confirm import (
    paper_search_confirm_node,
)
from app.llm.graph.workflows.paper_search.nodes.create_task import (
    CreatePaperSearchTaskNode,
)
from app.llm.graph.workflows.paper_search.nodes.generate_queries import (
    BuildSourceQueryPlanNode,
)
from app.llm.graph.workflows.paper_search.nodes.search import (
    MultiSourceSearchNode,
)
from app.llm.graph.workflows.paper_search.nodes.filter import (
    NormalizeDeduplicateFilterNode,
)
from app.llm.graph.workflows.paper_search.nodes.enrich import (
    PaperEnrichmentNode,
)
from app.llm.graph.workflows.paper_search.nodes.crossref_enrich import (
    CrossrefMetadataEnrichmentNode,
)
from app.llm.graph.workflows.paper_search.nodes.venue import (
    VenueResolutionNode,
)
from app.llm.graph.workflows.paper_search.nodes.dowload_pdf import (
    PdfDownloadNode,
)
from app.llm.graph.workflows.paper_search.nodes.abstract_enrich import (
    AbstractEnrichNode,
)
from app.llm.graph.workflows.paper_search.nodes.recommend import (
    RecommendationNode,
)
from app.llm.graph.workflows.paper_search.state import (
    PaperSearchWorkflowState,
)
from app.infrastructure.task_service_grpc_client import (
    TaskServiceGrpcClient,
)


class FakePaperServiceClient:
    async def add_query_task(self, request: dict) -> dict:
        return {
            "ok": True,
            "result": {
                "task_id": 10001,
                "status": "task_created",
                "request": request,
            },
            "error": None,
        }

    async def update_task_status(self, **_kwargs) -> dict:
        return {"ok": True, "result": {"updated": True}, "error": None}


def _route_after_confirm(state: PaperSearchWorkflowState) -> str:
    return (
        "create_task"
        if state.get("stage") == "creating_task"
        else END
    )


def _route_after_create_task(state: PaperSearchWorkflowState) -> str:
    return (
        "generate_queries"
        if state.get("stage") == "generating_source_queries"
        else END
    )


def _route_after_generate_queries(
    state: PaperSearchWorkflowState,
) -> str:
    return "search" if state.get("stage") == "searching" else END


def _route_after_search(state: PaperSearchWorkflowState) -> str:
    return "filter" if state.get("stage") == "normalizing" else END


def _route_after_filter(state: PaperSearchWorkflowState) -> str:
    return (
        "enrich"
        if state.get("stage") in {"enriching", "reviewing_search"}
        else END
    )


def _route_after_enrich(state: PaperSearchWorkflowState) -> str:
    return (
        "crossref_enrich"
        if state.get("stage") == "enriching_crossref"
        else END
    )


def _route_after_crossref_enrich(
    state: PaperSearchWorkflowState,
) -> str:
    return "venue" if state.get("stage") == "resolving_venues" else END


def _route_after_venue(state: PaperSearchWorkflowState) -> str:
    return "download_pdf" if state.get("stage") == "downloading_pdfs" else END


def _route_after_download_pdf(
    state: PaperSearchWorkflowState,
) -> str:
    return (
        "abstract_enrich"
        if state.get("stage") == "enriching_abstract"
        else END
    )


def _route_after_abstract_enrich(
    state: PaperSearchWorkflowState,
) -> str:
    return (
        "recommend"
        if state.get("stage") == "recommending"
        else END
    )


def build_intent_tag_confirm_task_query_search_filter_enrich_venue_subgraph(
    client,
):
    builder = StateGraph(PaperSearchWorkflowState)
    builder.add_node(
        "intent_understanding",
        IntentUnderstandingNode(),
    )
    builder.add_node(
        "build_search_tag",
        BuildDefaultSearchTagNode(),
    )
    builder.add_node(
        "confirm",
        paper_search_confirm_node,
    )
    builder.add_node(
        "create_task",
        CreatePaperSearchTaskNode(client=client),
    )
    builder.add_node(
        "generate_queries",
        BuildSourceQueryPlanNode(),
    )
    builder.add_node(
        "search",
        MultiSourceSearchNode(),
    )
    builder.add_node(
        "filter",
        NormalizeDeduplicateFilterNode(),
    )
    builder.add_node("enrich", PaperEnrichmentNode())
    builder.add_node("crossref_enrich", CrossrefMetadataEnrichmentNode())
    builder.add_node("venue", VenueResolutionNode())
    builder.add_node("download_pdf", PdfDownloadNode())
    builder.add_node("abstract_enrich", AbstractEnrichNode())
    builder.add_node("recommend", RecommendationNode())
    builder.add_edge(START, "intent_understanding")
    builder.add_edge("intent_understanding", "build_search_tag")
    builder.add_edge("build_search_tag", "confirm")
    builder.add_conditional_edges("confirm", _route_after_confirm)
    builder.add_conditional_edges(
        "create_task",
        _route_after_create_task,
    )
    builder.add_conditional_edges(
        "generate_queries",
        _route_after_generate_queries,
    )
    builder.add_conditional_edges("search", _route_after_search)
    builder.add_conditional_edges("filter", _route_after_filter)
    builder.add_conditional_edges("enrich", _route_after_enrich)
    builder.add_conditional_edges(
        "crossref_enrich",
        _route_after_crossref_enrich,
    )
    builder.add_conditional_edges("venue", _route_after_venue)
    builder.add_conditional_edges(
        "download_pdf",
        _route_after_download_pdf,
    )
    builder.add_conditional_edges(
        "abstract_enrich",
        _route_after_abstract_enrich,
    )
    builder.add_edge("recommend", END)
    return builder.compile(checkpointer=MemorySaver())


async def run_graph(client) -> None:
    graph = (
        build_intent_tag_confirm_task_query_search_filter_enrich_venue_subgraph(
            client
        )
    )
    config = {"configurable": {"thread_id": "intent-confirm-demo"}}
    run_id = f"paper-search-node-test-{uuid.uuid4().hex}"

    interrupted = await graph.ainvoke(
        {
            "run_id": run_id,
            "original_prompt": (
                "帮我找 2023 到 2026 年关于 RAG 评测的论文，"
                "重点关注 benchmark，排除医疗场景。"
            ),
        },
        config=config,
    )

    print("确认请求：")
    print(
        json.dumps(
            interrupted.get("__interrupt__", ())[0].value,
            ensure_ascii=False,
            indent=2,
        )
    )

    result = await graph.ainvoke(
        Command(resume={"decision": "approved"}),
        config=config,
    )

    print("\n确认后的 State：")
    print(
        json.dumps(
            {
                "query_understanding": result.get(
                    "query_understanding"
                ),
                "intent_error": result.get("intent_error"),
                "search_tag": result.get("search_tag"),
                "confirmation_decision": result.get(
                    "confirmation_decision"
                ),
                "confirmation_required": result.get(
                    "confirmation_required"
                ),
                "paper_service_task_id": result.get(
                    "paper_service_task_id"
                ),
                "create_task_payload": result.get(
                    "create_task_payload"
                ),
                "run_id": result.get("run_id"),
                "source_query_plans": result.get(
                    "source_query_plans"
                ),
                "source_summaries": result.get("source_summaries"),
                "raw_result_artifact_refs": result.get(
                    "raw_result_artifact_refs"
                ),
                "normalized_manifest_artifact_ref": result.get(
                    "normalized_manifest_artifact_ref"
                ),
                "existing_papers_manifest_artifact_ref": result.get(
                    "existing_papers_manifest_artifact_ref"
                ),
                "enrichment_manifest_artifact_ref": result.get(
                    "enrichment_manifest_artifact_ref"
                ),
                "crossref_enrichment_manifest_artifact_ref": result.get(
                    "crossref_enrichment_manifest_artifact_ref"
                ),
                "venue_manifest_artifact_ref": result.get(
                    "venue_manifest_artifact_ref"
                ),
                "pdf_manifest_artifact_ref": result.get(
                    "pdf_manifest_artifact_ref"
                ),
                "abstract_manifest_artifact_ref": result.get(
                    "abstract_manifest_artifact_ref"
                ),
                "recommendation_manifest_artifact_ref": result.get(
                    "recommendation_manifest_artifact_ref"
                ),
                "progress": result.get("progress"),
                "warnings": result.get("warnings"),
                "stage": result.get("stage"),
                "status": result.get("status"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

async def main() -> None:
    use_real_paper_service = (
        os.getenv("USE_REAL_PAPER_SERVICE", "false").lower()
        == "true"
    )

    if not use_real_paper_service:
        print("Test mode: fake paper-service client")
        await run_graph(FakePaperServiceClient())
        return

    print("Integration mode: real Nacos and paper-service")
    from app.main import app

    async with app.router.lifespan_context(app):
        await run_graph(TaskServiceGrpcClient())


if __name__ == "__main__":
    asyncio.run(main())
