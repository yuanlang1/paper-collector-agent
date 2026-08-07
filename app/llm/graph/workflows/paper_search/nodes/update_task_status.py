from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.infrastructure.grpc.task_service_grpc_client import (
    TaskServiceGrpcClient,
    TaskState,
    task_service_grpc_client,
)
from app.rag.processing.task_rag_batch_runner import (
    TaskRagBatchRunner,
    get_task_rag_batch_runner,
)


def _is_valid_task_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


class UpdatePaperSearchTaskStatusNode:

    def __init__(
        self,
        client: TaskServiceGrpcClient | None = None,
        rag_runner: TaskRagBatchRunner | None = None
    ) -> None:
        self.client = client or task_service_grpc_client
        self.rag_runner = rag_runner

    async def __call__(
        self,
        state: Mapping[str, Any]
    ) -> dict[str, Any]:
        task_id = state.get("paper_service_task_id")
        if not _is_valid_task_id(task_id):
            return {}

        stage = state.get("stage")
        status = state.get("status")
        if stage == "failed" or status in {"failed", "blocked"}:
            task_state = TaskState.SEARCH_FAILED
            error_message = str(state.get("error") or "paper search failed")

        elif stage == "partial_failed" or status == "partial_failed":
            task_state = TaskState.SEARCH_PARTIAL_COMPLETED
            error_message = None

        else:
            task_state = TaskState.SEARCH_COMPLETED
            error_message = None

        try:
            result = await self.client.update_task_status(
                task_id = task_id,
                task_state = task_state,
                error_message = error_message,
            )
        except Exception as exc:
            result = {
                "ok": False,
                "error": str(exc),
            }

        if result.get("ok") is True:
            rag_task_started = False
            if task_state is TaskState.SEARCH_COMPLETED:
                runner = self.rag_runner or get_task_rag_batch_runner()
                rag_task_started = await runner.notify(task_id)

            return {
                "progress": {
                    **state.get("progress", {}),
                    "task_status_update_attempted": 1,
                    "task_status_updated": 1,
                },
                "task_status_update_error": None,
                "remote_task_state": task_state.name,
                "rag_task_started": rag_task_started,
            }

        error = str(result.get("error") or "unknown error")
        stage = state.get("stage")
        status = state.get("status")

        if stage == "completed":
            stage = "partial_failed"
        if status == "completed":
            status = "partial_failed"
        return {
            "stage": stage,
            "status": status,
            "degraded": True,
            "progress": {
                **state.get("progress", {}),
                "task_status_update_attempted": 1,
                "task_status_updated": 0,
            },
            "warnings": [
                *state.get("warnings", []),
                f"Task status update failed: {error}",
            ],
            "task_status_update_error": error,
            "remote_task_state": None,
            "error": (
                state.get("error")
                or f"Task status update failed: {error}"
            ),
        }
