from typing import Any, Literal

from typing_extensions import TypedDict


class TaskIndexingOutputState(TypedDict):
    content: str
    status: Literal["success", "failed", "timed_out"]


class TaskIndexingWorkflowState(TypedDict, total=False):
    run_id: str
    child_run_id: str
    content: str

    task_id: int
    remote_task_state: str | None

    stage: Literal[
        "checking_status",
        "indexing",
        "verifying",
        "completed",
        "failed",
        "timed_out",
    ]
    status: Literal["running", "completed", "failed", "timed_out", "success"]
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
