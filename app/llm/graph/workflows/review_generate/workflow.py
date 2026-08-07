from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.infrastructure.grpc.paper_service_grpc_client import (
    PaperServiceGrpcClient,
)
from app.llm.graph.workflows.review_generate.nodes.assemble_review import (
    AssembleReviewNode,
)
from app.llm.graph.workflows.review_generate.nodes.finalizing_handoff import (
    FinalizingHandoffNode,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import (
    finalize_task_review_node,
)
from app.llm.graph.workflows.review_generate.nodes.generate_claim import (
    GenerateClaimsNode,
)
from app.llm.graph.workflows.review_generate.nodes.generate_framework import (
    GenerateFrameworkNode,
)
from app.llm.graph.workflows.review_generate.nodes.initialize import (
    initialize_review_node,
)
from app.llm.graph.workflows.review_generate.nodes.load import (
    LoadTaskCorpusNode,
)
from app.llm.graph.workflows.review_generate.nodes.reflect_review import (
    ReflectReviewNode,
)
from app.llm.graph.workflows.review_generate.nodes.render_section import (
    RenderSectionsNode,
)
from app.llm.graph.workflows.review_generate.nodes.retrieve_evidence import (
    RetrieveEvidenceNode,
)
from app.llm.graph.workflows.review_generate.state import (
    TaskReviewWorkflowState,
)


def _route(expected_stage: str, target: str):
    def route(state: TaskReviewWorkflowState) -> str:
        if state.get("stage") == expected_stage:
            return target
        return "finalize"

    return route


def _route_after_reflection(
    state: TaskReviewWorkflowState,
) -> str:
    return {
        "finalizing_handoff": "finalizing_handoff",
        "generating_claims": "generate_claims",
        "retrieving_evidence": "retrieve_evidence",
        "rendering_sections": "render_sections",
    }.get(state.get("stage"), "finalize")


def build_task_review_workflow(
    *,
    paper_client: PaperServiceGrpcClient | None = None,
    checkpointer: Any | None = None,
    node_overrides: dict[str, object] | None = None,
):
    node_overrides = node_overrides or {}

    def node(name: str, factory):
        if name in node_overrides:
            return node_overrides[name]
        return factory()

    builder = StateGraph(TaskReviewWorkflowState)
    builder.add_node("initialize", node("initialize", lambda: initialize_review_node))
    builder.add_node(
        "load_task_corpus",
        node(
            "load_task_corpus",
            lambda: LoadTaskCorpusNode(client=paper_client),
        ),
    )
    builder.add_node(
        "generate_framework",
        node("generate_framework", GenerateFrameworkNode),
    )
    builder.add_node(
        "generate_claims",
        node("generate_claims", GenerateClaimsNode),
    )
    builder.add_node(
        "retrieve_evidence",
        node("retrieve_evidence", RetrieveEvidenceNode),
    )
    builder.add_node(
        "render_sections",
        node("render_sections", RenderSectionsNode),
    )
    builder.add_node(
        "assemble_review",
        node("assemble_review", AssembleReviewNode),
    )
    builder.add_node(
        "reflect_review",
        node("reflect_review", ReflectReviewNode),
    )
    builder.add_node(
        "finalizing_handoff",
        node("finalizing_handoff", FinalizingHandoffNode),
    )
    builder.add_node(
        "finalize_task_review",
        node("finalize_task_review", lambda: finalize_task_review_node),
    )

    builder.add_edge(START, "initialize")
    builder.add_conditional_edges(
        "initialize",
        _route("loading_corpus", "load_task_corpus"),
        {
            "load_task_corpus": "load_task_corpus",
            "finalize": "finalize_task_review",
        },
    )
    builder.add_conditional_edges(
        "load_task_corpus",
        _route("generating_framework", "generate_framework"),
        {
            "generate_framework": "generate_framework",
            "finalize": "finalize_task_review",
        },
    )
    builder.add_conditional_edges(
        "generate_framework",
        _route("generating_claims", "generate_claims"),
        {
            "generate_claims": "generate_claims",
            "finalize": "finalize_task_review",
        },
    )
    builder.add_conditional_edges(
        "generate_claims",
        _route("retrieving_evidence", "retrieve_evidence"),
        {
            "retrieve_evidence": "retrieve_evidence",
            "finalize": "finalize_task_review",
        },
    )
    builder.add_conditional_edges(
        "retrieve_evidence",
        _route("rendering_sections", "render_sections"),
        {
            "render_sections": "render_sections",
            "finalize": "finalize_task_review",
        },
    )
    builder.add_conditional_edges(
        "render_sections",
        _route("assembling_review", "assemble_review"),
        {
            "assemble_review": "assemble_review",
            "finalize": "finalize_task_review",
        },
    )
    builder.add_conditional_edges(
        "assemble_review",
        _route("reflecting_review", "reflect_review"),
        {
            "reflect_review": "reflect_review",
            "finalize": "finalize_task_review",
        },
    )
    builder.add_conditional_edges(
        "reflect_review",
        _route_after_reflection,
        {
            "finalizing_handoff": "finalizing_handoff",
            "generate_claims": "generate_claims",
            "retrieve_evidence": "retrieve_evidence",
            "render_sections": "render_sections",
            "finalize": "finalize_task_review",
        },
    )
    builder.add_edge("finalizing_handoff", "finalize_task_review")
    builder.add_edge("finalize_task_review", END)

    return builder.compile(checkpointer=checkpointer)
