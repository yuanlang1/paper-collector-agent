from __future__ import annotations

from typing import Any

from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.infrastructure.paper_service.paper_client import PaperServiceClient
from app.llm.graph.workflows.progress import instrument_subagent_progress_node
from app.llm.graph.workflows.review_generate.contracts import review_batch
from app.llm.graph.workflows.review_generate.nodes.assemble_review import (
    AssembleReviewNode,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import (
    finalize_task_review_node,
)
from app.llm.graph.workflows.review_generate.nodes.finalizing_handoff import (
    FinalizingHandoffNode,
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
from app.llm.graph.workflows.review_generate.nodes.load import LoadTaskCorpusNode
from app.llm.graph.workflows.review_generate.nodes.persist_review import (
    PersistReviewNode,
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
from app.llm.graph.workflows.review_generate.nodes.verify_claims import (
    VerificationProtocolError,
    VerifyClaimsNode,
)
from app.llm.graph.workflows.review_generate.state import (
    TaskReviewOutputState,
    TaskReviewWorkflowState,
)

REVIEW_MAX_CONCURRENCY = 4

_BATCH_NODES = {
    "generate_claims": ("generate_claims_item", "collect_claims"),
    "retrieve_evidence": ("retrieve_evidence_item", "collect_evidence"),
    "verify_claims": ("verify_claim_item", "collect_verification"),
    "render_sections": ("render_section_item", "collect_sections"),
}


_PHASES = {
    "initialize": ("prepare", "初始化综述请求"),
    "load_task_corpus": ("corpus", "加载任务语料"),
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
    counts = {
        "papers": len(current.get("paper_ids_snapshot") or []),
        "warnings": len(current.get("warnings") or []),
    }
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


def _route_after_verification(
    state: TaskReviewWorkflowState,
) -> str:
    return {
        "generating_claims": "generate_claims",
        "rendering_sections": "render_sections",
    }.get(state.get("stage"), "finalize")


def _route_after_writing_review(
    state: TaskReviewWorkflowState,
) -> str:
    return {
        "finalizing_handoff": "finalizing_handoff",
        "rendering_sections": "render_sections",
        "assembling_review": "assemble_review",
    }.get(state.get("stage"), "finalize")


def _batch_error(exc):
    return {
        "error": str(exc) or type(exc).__name__,
        "protocol_error": isinstance(exc, VerificationProtocolError),
    }


def _add_review_batch(builder, name, stage, route, paths, *, instrument=None):
    item_node, collect_node = _BATCH_NODES[name]

    async def prepare(state):
        try:
            return await stage.prepare(state)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return {
                **review_batch([]),
                "batch_jobs": None,
                **stage.failure(state, _batch_error(exc)),
            }

    async def run_item(item):
        try:
            record = {"value": await stage.run_item(item)}
        except GraphBubbleUp:
            raise
        except Exception as exc:
            record = _batch_error(exc)
        return {"batch_results": [{"item_id": item["item_id"], **record}]}

    async def collect(state):
        records = state["batch_results"]
        results = {
            item["item_id"]: item["value"] for item in records if "value" in item
        }
        positions = {
            job["item_id"]: index for index, job in enumerate(state["batch_jobs"])
        }
        failure = min(
            (item for item in records if "error" in item),
            key=lambda item: positions.get(item["item_id"], -1),
            default=None,
        )
        try:
            update = await stage.collect(state, results, failure)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            update = stage.failure(state, _batch_error(exc))
        if "stage" in update:
            return {**review_batch([]), **update}
        return update

    async def dispatch(state):
        jobs = state["batch_jobs"]
        if jobs is None:
            return "finalize"
        return [Send(item_node, item) for item in jobs] or collect_node

    async def after_collect(state):
        if state["batch_jobs"]:
            return [Send(item_node, item) for item in state["batch_jobs"]]
        return route(state)

    if instrument is not None:
        prepare = instrument(name, prepare, emit_start=True)
        collect = instrument(name, collect, emit_start=False)
    builder.add_node(name, prepare)
    builder.add_node(item_node, run_item)
    builder.add_node(collect_node, collect)
    builder.add_conditional_edges(
        name,
        dispatch,
        {
            item_node: item_node,
            collect_node: collect_node,
            "finalize": paths["finalize"],
        },
    )
    builder.add_edge(item_node, collect_node)
    collect_paths = (
        {**paths, item_node: item_node} if name == "generate_claims" else paths
    )
    builder.add_conditional_edges(collect_node, after_collect, collect_paths)


def build_task_review_workflow(
    *,
    paper_client: PaperServiceClient | None = None,
    user_id: int,
    checkpointer: Any | None = None,
    node_overrides: dict[str, object] | None = None,
):
    node_overrides = node_overrides or {}

    def progress(name, target, **kwargs):
        phase, phase_label = _PHASES[name]
        return instrument_subagent_progress_node(
            node_name=name,
            node=target,
            phase=phase,
            phase_label=phase_label,
            data_builder=_progress_data,
            iteration_resolver=_iteration_resolver(name),
            **kwargs,
        )

    def node(name: str, factory):
        return progress(
            name, node_overrides[name] if name in node_overrides else factory()
        )

    def batch_progress(name, target, *, emit_start):
        return progress(
            name,
            target,
            emit_start=emit_start,
            finish_when=lambda update: "stage" in update,
        )

    builder = StateGraph(TaskReviewWorkflowState, output_schema=TaskReviewOutputState)
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
        node("generate_framework", lambda: GenerateFrameworkNode(user_id=user_id)),
    )
    builder.add_node(
        "assemble_review",
        node("assemble_review", lambda: AssembleReviewNode(user_id=user_id)),
    )
    builder.add_node(
        "reflect_review",
        node("reflect_review", lambda: ReflectReviewNode(user_id=user_id)),
    )
    builder.add_node(
        "finalizing_handoff",
        node("finalizing_handoff", FinalizingHandoffNode),
    )
    builder.add_node(
        "persist_review",
        node("persist_review", PersistReviewNode),
    )
    builder.add_node(
        "finalize_result",
        node("finalize_result", lambda: finalize_task_review_node),
    )

    builder.add_edge(START, "initialize")
    builder.add_conditional_edges(
        "initialize",
        _route("loading_corpus", "load_task_corpus"),
        {
            "load_task_corpus": "load_task_corpus",
            "finalize": "finalize_result",
        },
    )
    builder.add_conditional_edges(
        "load_task_corpus",
        _route("generating_framework", "generate_framework"),
        {"generate_framework": "generate_framework", "finalize": "finalize_result"},
    )
    for name, factory, route, paths in (
        (
            "generate_claims",
            lambda: GenerateClaimsNode(user_id=user_id),
            _route("retrieving_evidence", "retrieve_evidence"),
            {"retrieve_evidence": "retrieve_evidence", "finalize": "finalize_result"},
        ),
        (
            "retrieve_evidence",
            RetrieveEvidenceNode,
            _route("verifying_claims", "verify_claims"),
            {"verify_claims": "verify_claims", "finalize": "finalize_result"},
        ),
        (
            "verify_claims",
            lambda: VerifyClaimsNode(user_id=user_id),
            _route_after_verification,
            {
                "generate_claims": "generate_claims",
                "render_sections": "render_sections",
                "finalize": "finalize_result",
            },
        ),
        (
            "render_sections",
            lambda: RenderSectionsNode(user_id=user_id),
            _route("assembling_review", "assemble_review"),
            {"assemble_review": "assemble_review", "finalize": "finalize_result"},
        ),
    ):
        if name in node_overrides:
            builder.add_node(name, node(name, factory))
            builder.add_conditional_edges(name, route, paths)
        else:
            _add_review_batch(
                builder, name, factory(), route, paths, instrument=batch_progress
            )

    builder.add_conditional_edges(
        "generate_framework",
        _route("generating_claims", "generate_claims"),
        {"generate_claims": "generate_claims", "finalize": "finalize_result"},
    )
    builder.add_conditional_edges(
        "assemble_review",
        _route("reflecting_review", "reflect_review"),
        {
            "reflect_review": "reflect_review",
            "finalize": "finalize_result",
        },
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

    return builder.compile(checkpointer=checkpointer).with_config(
        recursion_limit=128,
        max_concurrency=REVIEW_MAX_CONCURRENCY,
    )
