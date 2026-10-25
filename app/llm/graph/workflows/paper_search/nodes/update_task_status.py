from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.infrastructure.paper_service.task_client import (
    TaskServiceClient,
    task_service_client,
)
from app.infrastructure.paper_service.types import TaskState


def _is_valid_task_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


class UpdatePaperSearchTaskStatusNode:

    def __init__(
        self,
        client: TaskServiceClient | None = None,
    ) -> None:
        self.client = client or task_service_client

    async def __call__(
        self,
        state: Mapping[str, Any]
    ) -> dict[str, Any]:
        task_id = state.get("paper_service_task_id")
        if not _is_valid_task_id(task_id):
            return {}

        stage = state.get("stage")
        status = state.get("status")
        if stage == "preparing_search" and status == "running":
            task_state = TaskState.SEARCH_RUNNING
            error_message = None
        elif stage == "failed" or status in {"failed", "blocked"}:
            task_state = TaskState.SEARCH_FAILED
            error_message = str(state.get("error") or "paper search failed")

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
            return {
                "progress": {
                    **state.get("progress", {}),
                    "task_status_update_attempted": 1,
                    "task_status_updated": 1,
                },
                "task_status_update_error": None,
                "remote_task_state": task_state.name,
            }

        error = str(result.get("error") or "unknown error")
        return {
            "stage": "failed",
            "status": "failed",
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
            "error": f"Task status update failed: {error}",
        }
