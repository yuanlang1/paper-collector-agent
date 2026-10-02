from langgraph.graph import END, START, StateGraph

from app.llm.graph.workflows.paper_search.nodes.abstract_enrich import AbstractEnrichNode
from app.llm.graph.workflows.paper_search.nodes.create_task import CreatePaperSearchTaskNode
from app.llm.graph.workflows.paper_search.nodes.crossref_enrich import CrossrefMetadataEnrichmentNode
from app.llm.graph.workflows.paper_search.nodes.dowload_pdf import PdfDownloadNode
from app.llm.graph.workflows.paper_search.nodes.enrich import PaperEnrichmentNode
from app.llm.graph.workflows.paper_search.nodes.finalize_pdfs import FinalizePdfsNode
from app.llm.graph.workflows.paper_search.nodes.filter import NormalizeDeduplicateFilterNode
from app.llm.graph.workflows.paper_search.nodes.initialize import initialize_paper_search_node
from app.llm.graph.workflows.paper_search.nodes.finalize import finalize_paper_search_node
from app.llm.graph.workflows.paper_search.nodes.plan_search import PlanSearchNode
from app.llm.graph.workflows.paper_search.nodes.persist import PersistRecommendedPapersNode
from app.llm.graph.workflows.paper_search.nodes.recommend import RecommendationNode
from app.llm.graph.workflows.paper_search.nodes.review import ReviewNode
from app.llm.graph.workflows.paper_search.nodes.search import SearchNode
from app.llm.graph.workflows.paper_search.nodes.update_task_status import UpdatePaperSearchTaskStatusNode
from app.llm.graph.workflows.paper_search.nodes.venue import VenueResolutionNode
from app.llm.graph.workflows.paper_search.state import PaperSearchWorkflowState
from app.llm.provider import ChatClient
from app.llm.graph.workflows.paper_search.progress import instrument_paper_search_node


def _route(stage: str, target: str):
    return lambda state: (
        target
        if state.get("stage") == stage
        else _terminal_target(state)
    )


def _route_paths(target: str) -> dict[str, str]:
    return {
        target: target,
        "finalize_pdfs": "finalize_pdfs",
        "finalize_result": "finalize_result",
    }


def _terminal_target(state: PaperSearchWorkflowState) -> str:
    if state.get("downloaded_pdf_paths"):
        return "finalize_pdfs"

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
        return "finalize_pdfs"
    return "finalize_result"


def _route_after_review(state: PaperSearchWorkflowState) -> str:
    if state.get("stage") == "searching":
        return "search"
    if state.get("stage") == "enriching":
        return "enrich"
    return _terminal_target(state)


def build_paper_search_workflow(
    *,
    task_client=None,
    chat: ChatClient | None = None,
    checkpointer=None,
    node_overrides: dict[str, object] | None = None,
):
    node_overrides = node_overrides or {}
    chat = chat or ChatClient()

    def node(name, default):
        return instrument_paper_search_node(
            node_name=name,
            node=node_overrides.get(name, default),
        )

    builder = StateGraph(PaperSearchWorkflowState)
    builder.add_node(
        "initialize",
        node("initialize", initialize_paper_search_node),
    )
    builder.add_node(
        "plan_search",
        node(
            "plan_search",
            PlanSearchNode(chat=chat),
        ),
    )
    builder.add_node("create_task", node("create_task", CreatePaperSearchTaskNode(client=task_client)))
    builder.add_node("search", node("search", SearchNode()))
    builder.add_node("filter", node("filter", NormalizeDeduplicateFilterNode()))
    builder.add_node("review", node("review", ReviewNode(chat=chat)))
    builder.add_node("enrich", node("enrich", PaperEnrichmentNode()))
    builder.add_node("crossref_enrich", node("crossref_enrich", CrossrefMetadataEnrichmentNode()))
    builder.add_node("venue", node("venue", VenueResolutionNode(chat=chat)))
    builder.add_node("download_pdf", node("download_pdf", PdfDownloadNode()))
    builder.add_node("abstract_enrich", node("abstract_enrich", AbstractEnrichNode(chat=chat)))
    builder.add_node("recommend", node("recommend", RecommendationNode(chat=chat)))
    builder.add_node("persist", node("persist", PersistRecommendedPapersNode()))
    builder.add_node(
        "finalize_pdfs",
        node(
            "finalize_pdfs",
            FinalizePdfsNode(),
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
        _route("preparing_search", "plan_search"),
        _route_paths("plan_search"),
    )
    builder.add_conditional_edges(
        "plan_search",
        _route("searching", "search"),
        _route_paths("search"),
    )
    builder.add_conditional_edges(
        "search",
        _route("normalizing", "filter"),
        _route_paths("filter"),
    )
    builder.add_conditional_edges(
        "filter",
        _route("reviewing_search", "review"),
        _route_paths("review"),
    )
    builder.add_conditional_edges(
        "review",
        _route_after_review,
        {
            "search": "search",
            "enrich": "enrich",
            "finalize_pdfs": "finalize_pdfs",
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
    builder.add_edge("persist", "finalize_pdfs")
    builder.add_edge("finalize_pdfs", "update_task_status")
    builder.add_edge("update_task_status", "finalize_result")
    builder.add_edge("finalize_result", END)
    return builder.compile(checkpointer=checkpointer)
