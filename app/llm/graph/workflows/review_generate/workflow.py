from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.infrastructure.grpc.paper_service_grpc_client import PaperServiceGrpcClient
from app.llm.graph.workflows.review_generate.nodes.assemble_review import AssembleReviewNode
from app.llm.graph.workflows.review_generate.nodes.finalizing_handoff import FinalizingHandoffNode
from app.llm.graph.workflows.review_generate.nodes.finalize import finalize_task_review_node
from app.llm.graph.workflows.review_generate.nodes.extract_studies import ExtractStudiesNode
from app.llm.graph.workflows.review_generate.nodes.generate_claim import GenerateClaimsNode
from app.llm.graph.workflows.review_generate.nodes.generate_framework import GenerateFrameworkNode
from app.llm.graph.workflows.review_generate.nodes.initialize import initialize_review_node
from app.llm.graph.workflows.review_generate.nodes.load import LoadTaskCorpusNode
from app.llm.graph.workflows.review_generate.nodes.persist_review import PersistReviewNode
from app.llm.graph.workflows.review_generate.nodes.reflect_review import ReflectReviewNode
from app.llm.graph.workflows.review_generate.nodes.render_section import RenderSectionsNode
from app.llm.graph.workflows.review_generate.nodes.retrieve_evidence import RetrieveEvidenceNode
from app.llm.graph.workflows.review_generate.state import TaskReviewWorkflowState
from app.llm.graph.workflows.progress import instrument_subagent_progress_node

from app.llm.graph.workflows.review_generate.nodes.verify_claims import VerifyClaimsNode
from app.llm.provider import ChatClient


_PHASES = {
    "initialize": ("prepare", "初始化综述请求"),
    "load_task_corpus": ("corpus", "加载任务语料"),
    "extract_studies": ("reading", "提取论文研究信息"),
    "generate_framework": ("framework", "生成综述框架"),
    "generate_claims": ("claims", "生成或修订论点"),
    "retrieve_evidence": ("evidence", "检索或补充证据"),
    "verify_claims": ("verify", "核验论点"),
    "render_sections": ("writing", "撰写或修订章节"),
    "assemble_review": ("writing", "组装综述"),
    "reflect_review": ("review", "反思与质量检查"),
    "finalizing_handoff": ("persist", "整理交接结果"),
    "persist_review": ("persist", "保存综述"),
    "finalize_result": ("finalize", "汇总综述结果"),
}

_CLAIM_LOOP_NODES = {
    "generate_claims",
    "retrieve_evidence",
    "verify_claims",
}

_WRITING_LOOP_NODES = {
    "render_sections",
    "assemble_review",
    "reflect_review",
}


def _iteration_resolver(node_name: str):
    if node_name in _CLAIM_LOOP_NODES:
        return lambda state: int(state.get("reflection_round", 0) or 0) + 1
    if node_name in _WRITING_LOOP_NODES:
        return lambda state: int(state.get("writing_revision_round", 0) or 0) + 1
    return None


def _progress_data(state, update):
    current = {**state, **(update or {})}
    reading_summary = current.get("reading_summary")
    counts = {
        "papers": len(current.get("paper_ids_snapshot") or []),
        "warnings": len(current.get("warnings") or []),
    }
    if isinstance(reading_summary, dict):
        counts.update({
            "profiles": int(reading_summary.get("succeeded", 0)),
            "failed": int(reading_summary.get("failed", 0)),
        })
    return {
        "task_id": current.get("task_id"),
        "counts": counts,
    }


def _route(expected_stage: str, target: str):
    def route(state: TaskReviewWorkflowState) -> str:
        if state.get("stage") == expected_stage:
            return target
        return "finalize"

    return route


def _route_after_verification(state: TaskReviewWorkflowState,) -> str:
    return {
        "generating_claims": "generate_claims",
        "rendering_sections": "render_sections",
    }.get(state.get("stage"), "finalize")


def _route_after_writing_review(state: TaskReviewWorkflowState,) -> str:
    return {
        "finalizing_handoff": "finalizing_handoff",
        "rendering_sections": "render_sections",
        "assembling_review": "assemble_review",
    }.get(state.get("stage"), "finalize")


def build_task_review_workflow(
    *,
    paper_client: PaperServiceGrpcClient | None = None,
    chat: ChatClient | None = None,
    checkpointer: Any | None = None,
    node_overrides: dict[str, object] | None = None,
):
    node_overrides = node_overrides or {}
    chat = chat or ChatClient()

    def node(name: str, factory):
        if name in node_overrides:
            target = node_overrides[name]
        else:
            target = factory()
        phase, phase_label = _PHASES[name]
        return instrument_subagent_progress_node(
            node_name=name,
            node=target,
            phase=phase,
            phase_label=phase_label,
            data_builder=_progress_data,
            iteration_resolver=_iteration_resolver(name),
        )

    builder = StateGraph(TaskReviewWorkflowState)
    builder.add_node("verify_claims", node("verify_claims", lambda: VerifyClaimsNode(chat=chat)))
    builder.add_node("initialize", node("initialize", lambda: initialize_review_node))
    builder.add_node(
        "load_task_corpus",
        node("load_task_corpus", lambda: LoadTaskCorpusNode(client=paper_client),),
    )
    builder.add_node(
        "generate_framework", node("generate_framework", lambda: GenerateFrameworkNode(chat=chat)),
    )
    builder.add_node(
        "extract_studies", node("extract_studies", lambda: ExtractStudiesNode(chat=chat)),
    )
    builder.add_node(
        "generate_claims", node("generate_claims", lambda: GenerateClaimsNode(chat=chat)),
    )
    builder.add_node(
        "retrieve_evidence", node("retrieve_evidence", RetrieveEvidenceNode),
    )
    builder.add_node(
        "render_sections", node("render_sections", lambda: RenderSectionsNode(chat=chat)),
    )
    builder.add_node(
        "assemble_review", node("assemble_review", lambda: AssembleReviewNode(chat=chat)),
    )
    builder.add_node(
        "reflect_review",
        node("reflect_review", lambda: ReflectReviewNode(chat=chat)),
    )
    builder.add_node(
        "finalizing_handoff", node("finalizing_handoff", FinalizingHandoffNode),
    )
    builder.add_node(
        "persist_review", node("persist_review", PersistReviewNode),
    )
    builder.add_node(
        "finalize_result", node("finalize_result", lambda: finalize_task_review_node),
    )

    builder.add_edge(START, "initialize")
    builder.add_conditional_edges(
        "initialize",
        _route("loading_corpus", "load_task_corpus"),
        {"load_task_corpus": "load_task_corpus", "finalize": "finalize_result",},
    )
    builder.add_conditional_edges(
        "load_task_corpus",
        _route("extracting_studies", "extract_studies"),
        {"extract_studies": "extract_studies", "finalize": "finalize_result",},
    )
    builder.add_conditional_edges(
        "extract_studies",
        _route("generating_framework", "generate_framework"),
        {"generate_framework": "generate_framework", "finalize": "finalize_result"},
    )
    builder.add_conditional_edges(
        "generate_claims",
        _route("retrieving_evidence", "retrieve_evidence"),
        {"retrieve_evidence": "retrieve_evidence", "finalize": "finalize_result",},
    )
    builder.add_conditional_edges(
        "retrieve_evidence",
        _route("verifying_claims", "verify_claims"),
        {"verify_claims": "verify_claims", "finalize": "finalize_result"},
    )
    builder.add_conditional_edges(
        "verify_claims",
        _route_after_verification,
        {
            "generate_claims": "generate_claims",
            "render_sections": "render_sections",
            "finalize": "finalize_result",
        },
    )
    builder.add_conditional_edges(
        "generate_framework",
        _route("generating_claims", "generate_claims"),
        {"generate_claims": "generate_claims", "finalize": "finalize_result"},
    )
    builder.add_conditional_edges(
        "render_sections",
        _route("assembling_review", "assemble_review"),
        {"assemble_review": "assemble_review", "finalize": "finalize_result",},
    )
    builder.add_conditional_edges(
        "assemble_review",
        _route("reflecting_review", "reflect_review"),
        {"reflect_review": "reflect_review", "finalize": "finalize_result",},
    )
    builder.add_conditional_edges(
        "reflect_review",
        _route_after_writing_review,
        {
            "finalizing_handoff": "finalizing_handoff",
            "assemble_review": "assemble_review",
            "render_sections": "render_sections",
            "finalize": "finalize_result",
        },
    )
    builder.add_conditional_edges(
        "finalizing_handoff",
        _route("persisting_review", "persist_review"),
        {"persist_review": "persist_review", "finalize": "finalize_result"},
    )
    builder.add_edge("persist_review", "finalize_result")
    builder.add_edge("finalize_result", END)

    return builder.compile(checkpointer=checkpointer).with_config(recursion_limit=128)
