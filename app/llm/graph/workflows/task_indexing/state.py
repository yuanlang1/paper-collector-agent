from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class TaskIndexingWorkflowState(TypedDict, total=False):
    run_id: str
    active_tool_call: dict[str, Any] | None
    messages: Annotated[list[BaseMessage], add_messages]
    last_action_result: dict[str, Any] | None
    artifact_refs: list[str]

    task_id: int
    timeout_seconds: int
    poll_interval_seconds: int
    remote_task_state: str | None

    stage: Literal[
        "checking_status",
        "indexing",
        "verifying",
        "completed",
        "failed",
        "timed_out",
    ]
    status: Literal["running", "completed", "failed", "timed_out"]
    error_code: str | None
    error: str | None

    rag_status: dict[str, Any] | None
    overall_summary: dict[str, int]
    committed_summary: dict[str, int]
    current_batch_summary: dict[str, int]
    overall_progress_percent: int
    committed_progress_percent: int
    worker_result: dict[str, Any] | None
    warnings: list[str]
