from datetime import UTC, datetime

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from app.bus.events import SubagentProgressEvent
from app.llm.context import PaperSearchContext
from app.llm.graph.workflows.paper_search.nodes.collect_search import CollectSearchNode
from app.llm.graph.workflows.paper_search.nodes.create_task import (
    CreatePaperSearchTaskNode,
)
from app.llm.graph.workflows.paper_search.nodes.enrich import (
    CollectEnrichNode,
    PaperEnrichmentNode,
)
from app.llm.graph.workflows.paper_search.nodes.filter import (
    NormalizeDeduplicateFilterNode,
)
from app.llm.graph.workflows.paper_search.nodes.finalize import (
    finalize_paper_search_node,
)
from app.llm.graph.workflows.paper_search.nodes.initialize import (
    initialize_paper_search_node,
)
from app.llm.graph.workflows.paper_search.nodes.persist import (
    PersistRecommendedPapersNode,
)
from app.llm.graph.workflows.paper_search.nodes.plan_search import PlanSearchNode
from app.llm.graph.workflows.paper_search.nodes.probe_pdf import PdfProbeNode
from app.llm.graph.workflows.paper_search.nodes.recommend import (
    CollectRecommendationNode,
    PrepareRecommendationNode,
    RecommendationNode,
)
from app.llm.graph.workflows.paper_search.nodes.review import ReviewNode
from app.llm.graph.workflows.paper_search.nodes.search import SearchNode
from app.llm.graph.workflows.paper_search.nodes.update_task_status import (
    UpdatePaperSearchTaskStatusNode,
)
from app.llm.graph.workflows.paper_search.progress import instrument_paper_search_node
from app.llm.graph.workflows.paper_search.state import (
    EnrichWorkerState,
    PaperSearchOutputState,
    PaperSearchWorkflowState,
    RecommendWorkerState,
    SearchWorkerState,
)
from app.llm.graph.workflows.progress import instrument_subagent_progress_node


def _route(stage: str, target: str):
    return lambda state: (
        target if state.get("stage") == stage else _terminal_target(state)
    )


def _route_paths(target: str) -> dict[str, str]:
    return {
        target: target,
        "update_task_status": "update_task_status",
        "finalize_result": "finalize_result",
    }


def _terminal_target(state: PaperSearchWorkflowState) -> str:
    terminal = {"completed", "failed", "blocked"}
    if _has_existing_task(state) and (
        state.get("stage") in terminal or state.get("status") in terminal
    ):
        return "update_task_status"
    return "finalize_result"


def _dispatch_search(state):
    tasks = state.get("active_search_tasks", {})
    if not tasks:
        return _terminal_target(state)
    return [
        Send("search", {"search_task": task, "child_run_id": state["child_run_id"]})
        for task in tasks.values()
    ]


def _route_after_plan(state):
    return (
        _dispatch_search(state)
        if state.get("stage") == "searching"
        else _terminal_target(state)
    )


def _route_after_review(state):
    if state.get("stage") == "searching":
        return _dispatch_search(state)
    if state.get("stage") == "enriching":
        return "probe_pdf"
    return _terminal_target(state)


def _dispatch_enrich(
    state: PaperSearchWorkflowState, runtime: Runtime[PaperSearchContext]
):
    if state.get("stage") != "enriching":
        return _terminal_target(state)
    if not state["enrich_candidates"]:
        return "collect_enrich"
    runtime.stream_writer(
        SubagentProgressEvent(
            **runtime.context["event_fields"],
            ts=datetime.now(UTC).isoformat(),
            description="开始：补充论文信息。",
        )
    )
    return [
        Send(
            "enrich",
            {
                "paper_index": index,
                "paper": paper,
                "child_run_id": state["child_run_id"],
            },
        )
        for index, paper in enumerate(state["enrich_candidates"])
    ]


def _dispatch_recommend(state: PaperSearchWorkflowState):
    if state.get("stage") != "recommending":
        return _terminal_target(state)
    return [
        Send(
            "recommend",
            {
                "paper_index": index,
                "paper": paper,
                "query_understanding": state["query_understanding"],
            },
        )
        for index, paper in enumerate(state["recommendation_candidates"])
    ] or "collect_recommend"


def _has_existing_task(state: PaperSearchWorkflowState) -> bool:
    task_id = state.get("paper_service_task_id")
    return isinstance(task_id, int) and not isinstance(task_id, bool) and task_id > 0


def _route_after_initialize(state: PaperSearchWorkflowState) -> str:
    if state.get("stage") != "preparing_search":
        return _terminal_target(state)
    return "update_task_status" if _has_existing_task(state) else "plan_search"


def _route_after_task_status(state: PaperSearchWorkflowState) -> str:
    if (
        state.get("stage") == "preparing_search"
        and state.get("status") == "running"
        and state.get("remote_task_state") == "SEARCH_RUNNING"
    ):
        return "plan_search"
    return "finalize_result"


def build_paper_search_workflow(
    *,
    model: BaseChatModel,
    task_client=None,
    checkpointer=None,
    node_overrides: dict[str, object] | None = None,
):
    node_overrides = node_overrides or {}

    def node(name, default):
        return instrument_paper_search_node(
            node_name=name,
            node=node_overrides.get(name, default),
        )

    builder = StateGraph(
        PaperSearchWorkflowState,
        output_schema=PaperSearchOutputState,
        context_schema=PaperSearchContext,
    )
    builder.add_node(
        "initialize",
        node("initialize", initialize_paper_search_node),
    )
    builder.add_node(
        "plan_search",
        node(
            "plan_search",
            PlanSearchNode(model=model),
        ),
    )
    builder.add_node(
        "create_task",
        node("create_task", CreatePaperSearchTaskNode(client=task_client)),
    )
    builder.add_node(
        "search", node("search", SearchNode()), input_schema=SearchWorkerState
    )
    builder.add_node("collect_search", node("collect_search", CollectSearchNode()))
    builder.add_node("filter", node("filter", NormalizeDeduplicateFilterNode()))
    builder.add_node("review", node("review", ReviewNode(model=model)))
    builder.add_node("probe_pdf", node("probe_pdf", PdfProbeNode()))
    builder.add_node(
        "enrich",
        node_overrides.get("enrich", PaperEnrichmentNode()),
        input_schema=EnrichWorkerState,
    )
    builder.add_node(
        "collect_enrich",
        instrument_subagent_progress_node(
            node_name="collect_enrich",
            node=node_overrides.get("collect_enrich", CollectEnrichNode()),
            phase="enrich",
            phase_label="补充论文信息",
            emit_start=False,
            finish_when=lambda result: bool(
                result.get("error") or result.get("progress", {}).get("pdf_probe_passed")
            ),
        ),
    )
    builder.add_node(
        "prepare_recommend",
        node("prepare_recommend", PrepareRecommendationNode()),
    )
    builder.add_node(
        "recommend",
        node_overrides.get("recommend", RecommendationNode(model=model)),
        input_schema=RecommendWorkerState,
    )
    builder.add_node(
        "collect_recommend",
        node("collect_recommend", CollectRecommendationNode()),
    )
    builder.add_node("persist", node("persist", PersistRecommendedPapersNode()))
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
        _route_after_initialize,
        {
            **_route_paths("plan_search"),
        },
    )
    builder.add_conditional_edges(
        "plan_search",
        _route_after_plan,
        _route_paths("search"),
    )
    builder.add_edge("search", "collect_search")
    builder.add_conditional_edges(
        "collect_search",
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
            **_route_paths("probe_pdf"),
        },
    )
    builder.add_conditional_edges(
        "probe_pdf",
        _dispatch_enrich,
        {"enrich": "enrich", **_route_paths("collect_enrich")},
    )
    builder.add_edge("enrich", "collect_enrich")
    builder.add_conditional_edges(
        "collect_enrich",
        _route("recommending", "prepare_recommend"),
        _route_paths("prepare_recommend"),
    )
    builder.add_conditional_edges(
        "prepare_recommend",
        _dispatch_recommend,
        {"recommend": "recommend", **_route_paths("collect_recommend")},
    )
    builder.add_edge("recommend", "collect_recommend")
    builder.add_conditional_edges(
        "collect_recommend",
        _route("persisting_papers", "create_task"),
        _route_paths("create_task"),
    )
    builder.add_conditional_edges(
        "create_task",
        _route("persisting_papers", "persist"),
        _route_paths("persist"),
    )
    builder.add_edge("persist", "update_task_status")
    builder.add_conditional_edges(
        "update_task_status",
        _route_after_task_status,
        {
            "plan_search": "plan_search",
            "finalize_result": "finalize_result",
        },
    )
    builder.add_edge("finalize_result", END)
    return builder.compile(checkpointer=checkpointer)
