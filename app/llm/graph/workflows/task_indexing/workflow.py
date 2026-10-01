from langgraph.graph import END, START, StateGraph

from app.llm.graph.workflows.task_indexing.nodes.check_rag_status import (
    check_task_rag_status_node,
)
from app.llm.graph.workflows.task_indexing.nodes.finalize import (
    finalize_task_indexing_node,
)
from app.llm.graph.workflows.task_indexing.nodes.initialize import (
    initialize_task_indexing_node,
)
from app.llm.graph.workflows.task_indexing.nodes.run_or_wait import (
    RunOrWaitTaskRagNode,
)
from app.llm.graph.workflows.task_indexing.nodes.verify_completion import (
    verify_task_indexing_node,
)
from app.llm.graph.workflows.task_indexing.state import TaskIndexingWorkflowState
from app.llm.graph.workflows.progress import (
    instrument_subagent_progress_node,
)


_PHASES = {
    "initialize": ("prepare", "初始化索引请求"),
    "check_rag_status": ("prepare", "检查任务索引状态"),
    "run_or_wait": ("index", "索引论文到知识库"),
    "verify_completion": ("verify", "校验索引结果"),
    "finalize_result": ("finalize", "汇总索引结果"),
}


def _progress_data(state, update):
    current = {**state, **(update or {})}
    return {
        "task_id": current.get("task_id"),
        "counts": dict(current.get("overall_summary") or {}),
        "committed_counts": dict(current.get("committed_summary") or {}),
    }


def _route(expected_stage: str, target: str):
    return lambda state: target if state.get("stage") == expected_stage else "finalize_result"


def build_task_indexing_workflow(
    *,
    checkpointer=None,
    node_overrides: dict[str, object] | None = None,
):
    node_overrides = node_overrides or {}

    def node(name: str, default):
        phase, phase_label = _PHASES[name]
        return instrument_subagent_progress_node(
            node_name=name,
            node=node_overrides.get(name, default),
            phase=phase,
            phase_label=phase_label,
            data_builder=_progress_data,
        )

    builder = StateGraph(TaskIndexingWorkflowState)
    builder.add_node("initialize", node("initialize", initialize_task_indexing_node))
    builder.add_node(
        "check_rag_status",
        node("check_rag_status", check_task_rag_status_node),
    )
    builder.add_node(
        "run_or_wait",
        node("run_or_wait", RunOrWaitTaskRagNode()),
    )
    builder.add_node(
        "verify_completion",
        node("verify_completion", verify_task_indexing_node),
    )
    builder.add_node(
        "finalize_result",
        node("finalize_result", finalize_task_indexing_node),
    )

    builder.add_edge(START, "initialize")
    builder.add_conditional_edges(
        "initialize",
        _route("checking_status", "check_rag_status"),
        {"check_rag_status": "check_rag_status", "finalize_result": "finalize_result"},
    )
    builder.add_conditional_edges(
        "check_rag_status",
        _route("indexing", "run_or_wait"),
        {"run_or_wait": "run_or_wait", "finalize_result": "finalize_result"},
    )
    builder.add_conditional_edges(
        "run_or_wait",
        _route("verifying", "verify_completion"),
        {"verify_completion": "verify_completion", "finalize_result": "finalize_result"},
    )
    builder.add_edge("verify_completion", "finalize_result")
    builder.add_edge("finalize_result", END)
    return builder.compile(checkpointer=checkpointer)
