from langgraph.graph import END, START, StateGraph

from app.llm.graph.workflows.task_indexing.nodes import (
    RunOrWaitTaskRagNode,
    check_task_rag_status_node,
    finalize_task_indexing_node,
    initialize_task_indexing_node,
    verify_task_indexing_node,
)
from app.llm.graph.workflows.task_indexing.state import TaskIndexingWorkflowState
from app.llm.streaming.timeline import (
    TASK_INDEXING_TIMELINE,
    instrument_timeline_node,
)


def _route(expected_stage: str, target: str):
    return lambda state: target if state.get("stage") == expected_stage else "finalize_result"


def build_task_indexing_workflow(
    *,
    checkpointer=None,
    node_overrides: dict[str, object] | None = None,
):
    node_overrides = node_overrides or {}

    def node(name: str, default):
        return instrument_timeline_node(
            workflow="task_indexing",
            node_name=name,
            node=node_overrides.get(name, default),
            timeline=TASK_INDEXING_TIMELINE,
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
