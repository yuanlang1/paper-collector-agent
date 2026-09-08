from langgraph.graph import END, START, StateGraph

from app.llm.graph.workflows.paper_search.nodes.abstract_enrich import AbstractEnrichNode
from app.llm.graph.workflows.paper_search.nodes.build_search_tag import BuildSearchTagNode
from app.llm.graph.workflows.paper_search.nodes.cleanup import CleanupDownloadedPdfsNode
from app.llm.graph.workflows.paper_search.nodes.confirm import paper_search_confirm_node
from app.llm.graph.workflows.paper_search.nodes.create_task import CreatePaperSearchTaskNode
from app.llm.graph.workflows.paper_search.nodes.crossref_enrich import CrossrefMetadataEnrichmentNode
from app.llm.graph.workflows.paper_search.nodes.dowload_pdf import PdfDownloadNode
from app.llm.graph.workflows.paper_search.nodes.enrich import PaperEnrichmentNode
from app.llm.graph.workflows.paper_search.nodes.filter import NormalizeDeduplicateFilterNode
from app.llm.graph.workflows.paper_search.nodes.finalize_search import (
    finalize_source_search_node,
)
from app.llm.graph.workflows.paper_search.nodes.generate_queries import BuildSourceQueryPlanNode
from app.llm.graph.workflows.paper_search.nodes.initialize import initialize_paper_search_node
from app.llm.graph.workflows.paper_search.nodes.intent import IntentUnderstandingNode
from app.llm.graph.workflows.paper_search.nodes.finalize import finalize_paper_search_node
from app.llm.graph.workflows.paper_search.nodes.persist import PersistRecommendedPapersNode
from app.llm.graph.workflows.paper_search.nodes.recommend import RecommendationNode
from app.llm.graph.workflows.paper_search.nodes.search_review import SearchReviewBrainNode
from app.llm.graph.workflows.paper_search.nodes.source_search import ArxivSearchNode, DblpSearchNode, GoogleScholarSearchNode
from app.llm.graph.workflows.paper_search.nodes.supplemental_search import SupplementalSearchPlannerNode
from app.llm.graph.workflows.paper_search.nodes.update_task_status import UpdatePaperSearchTaskStatusNode
from app.llm.graph.workflows.paper_search.nodes.venue import VenueResolutionNode
from app.llm.graph.workflows.paper_search.state import PaperSearchWorkflowState
from app.llm.streaming.timeline import (
    PAPER_SEARCH_TIMELINE,
    instrument_timeline_node,
)


def _route(stage: str, target: str):
    return lambda state: (
        target
        if state.get("stage") == stage
        else _terminal_target(state)
    )


def _route_paths(target: str) -> dict[str, str]:
    return {
        target: target,
        "cleanup_downloaded_pdfs": "cleanup_downloaded_pdfs",
        "finalize_result": "finalize_result",
    }


def _terminal_target(state: PaperSearchWorkflowState) -> str:
    if state.get("downloaded_pdf_paths"):
        return "cleanup_downloaded_pdfs"

    task_id = state.get("paper_service_task_id")
    if (
        isinstance(task_id, int)
        and not isinstance(task_id, bool)
        and task_id > 0
        and (
            state.get("stage") in {
                "completed",
                "partial_failed",
                "failed",
                "blocked",
            }
            or state.get("status") in {
                "completed",
                "partial_failed",
                "failed",
                "blocked",
            }
        )
    ):
        return "cleanup_downloaded_pdfs"
    return "finalize_result"


def _route_after_search_review(state: PaperSearchWorkflowState) -> str:
    if state.get("stage") == "planning_supplemental_search":
        return "supplemental_search"
    if state.get("stage") == "enriching":
        return "enrich"
    return _terminal_target(state)


def _route_after_supplemental_search(
    state: PaperSearchWorkflowState,
) -> str:
    if state.get("stage") == "searching":
        return "search_arxiv"
    if state.get("stage") == "enriching":
        return "enrich"
    return _terminal_target(state)


def build_paper_search_workflow(
    *,
    task_client=None,
    checkpointer=None,
    skip_confirmation: bool = False,
    node_overrides: dict[str, object] | None = None,
):
    node_overrides = node_overrides or {}

    def node(name, default):
        return instrument_timeline_node(
            workflow="paper_search",
            node_name=name,
            node=node_overrides.get(name, default),
            timeline=PAPER_SEARCH_TIMELINE,
            round_key="supplemental_search_round",
        )

    builder = StateGraph(PaperSearchWorkflowState)
    builder.add_node(
        "initialize",
        node("initialize", initialize_paper_search_node),
    )
    builder.add_node("intent_understanding", node("intent_understanding", IntentUnderstandingNode()))
    builder.add_node(
        "build_search_tag",
        node(
            "build_search_tag",
            BuildSearchTagNode(
                skip_confirmation=skip_confirmation,
            ),
        ),
    )
    builder.add_node("confirm", node("confirm", paper_search_confirm_node))
    builder.add_node("create_task", node("create_task", CreatePaperSearchTaskNode(client=task_client)))
    builder.add_node("generate_queries", node("generate_queries", BuildSourceQueryPlanNode()))
    builder.add_node("search_arxiv", node("search_arxiv", ArxivSearchNode()))
    builder.add_node("search_dblp", node("search_dblp", DblpSearchNode()))
    builder.add_node("search_google", node("search_google", GoogleScholarSearchNode()))
    builder.add_node(
        "finalize_source_search",
        node(
            "finalize_source_search",
            finalize_source_search_node,
        ),
    )
    builder.add_node("filter", node("filter", NormalizeDeduplicateFilterNode()))
    builder.add_node("search_review", node("search_review", SearchReviewBrainNode()))
    builder.add_node("supplemental_search", node("supplemental_search", SupplementalSearchPlannerNode()))
    builder.add_node("enrich", node("enrich", PaperEnrichmentNode()))
    builder.add_node("crossref_enrich", node("crossref_enrich", CrossrefMetadataEnrichmentNode()))
    builder.add_node("venue", node("venue", VenueResolutionNode()))
    builder.add_node("download_pdf", node("download_pdf", PdfDownloadNode()))
    builder.add_node("abstract_enrich", node("abstract_enrich", AbstractEnrichNode()))
    builder.add_node("recommend", node("recommend", RecommendationNode()))
    builder.add_node("persist", node("persist", PersistRecommendedPapersNode()))
    builder.add_node(
        "cleanup_downloaded_pdfs",
        node(
            "cleanup_downloaded_pdfs",
            CleanupDownloadedPdfsNode(),
        ),
    )
    builder.add_node(
        "update_task_status",
        node(
            "update_task_status",
            UpdatePaperSearchTaskStatusNode(
                client=task_client,
            ),
        ),
    )
    builder.add_node(
        "finalize_result",
        node("finalize_result", finalize_paper_search_node),
    )
    builder.add_edge(START, "initialize")
    builder.add_conditional_edges(
        "initialize",
        _route("intent_understanding", "intent_understanding"),
        _route_paths("intent_understanding"),
    )
    builder.add_conditional_edges(
        "intent_understanding",
        _route("building_search_tag", "build_search_tag"),
        _route_paths("build_search_tag"),
    )
    if skip_confirmation:
        builder.add_conditional_edges(
            "build_search_tag",
            _route("generating_source_queries", "generate_queries"),
            _route_paths("generate_queries"),
        )
    else:
        builder.add_conditional_edges(
            "build_search_tag",
            _route("awaiting_confirmation", "confirm"),
            _route_paths("confirm"),
        )
        builder.add_conditional_edges(
            "confirm",
            _route("generating_source_queries", "generate_queries"),
            _route_paths("generate_queries"),
        )
    builder.add_conditional_edges(
        "generate_queries",
        _route("searching", "search_arxiv"),
        _route_paths("search_arxiv"),
    )
    builder.add_edge("search_arxiv", "search_dblp")
    builder.add_edge("search_dblp", "search_google")
    builder.add_edge("search_google", "finalize_source_search")
    builder.add_conditional_edges(
        "finalize_source_search",
        _route("normalizing", "filter"),
        _route_paths("filter"),
    )
    builder.add_conditional_edges(
        "filter",
        _route("reviewing_search", "search_review"),
        _route_paths("search_review"),
    )
    builder.add_conditional_edges(
        "search_review",
        _route_after_search_review,
        {
            "supplemental_search": "supplemental_search",
            "enrich": "enrich",
            "cleanup_downloaded_pdfs": "cleanup_downloaded_pdfs",
            "finalize_result": "finalize_result",
        },
    )
    builder.add_conditional_edges(
        "supplemental_search",
        _route_after_supplemental_search,
        {
            "search_arxiv": "search_arxiv",
            "enrich": "enrich",
            "cleanup_downloaded_pdfs": "cleanup_downloaded_pdfs",
            "finalize_result": "finalize_result",
        },
    )
    builder.add_conditional_edges(
        "enrich",
        _route("enriching_crossref", "crossref_enrich"),
        _route_paths("crossref_enrich"),
    )
    builder.add_conditional_edges(
        "crossref_enrich",
        _route("resolving_venues", "venue"),
        _route_paths("venue"),
    )
    builder.add_conditional_edges(
        "venue",
        _route("downloading_pdfs", "download_pdf"),
        _route_paths("download_pdf"),
    )
    builder.add_conditional_edges(
        "download_pdf",
        _route("enriching_abstract", "abstract_enrich"),
        _route_paths("abstract_enrich"),
    )
    builder.add_conditional_edges(
        "abstract_enrich",
        _route("recommending", "recommend"),
        _route_paths("recommend"),
    )
    builder.add_conditional_edges(
        "recommend",
        _route("persisting_papers", "create_task"),
        _route_paths("create_task"),
    )
    builder.add_conditional_edges(
        "create_task",
        _route("persisting_papers", "persist"),
        _route_paths("persist"),
    )
    builder.add_edge("persist", "cleanup_downloaded_pdfs")
    builder.add_edge("cleanup_downloaded_pdfs", "update_task_status")
    builder.add_edge("update_task_status", "finalize_result")
    builder.add_edge("finalize_result", END)
    return builder.compile(checkpointer=checkpointer)
