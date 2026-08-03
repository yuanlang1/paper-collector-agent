from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.infrastructure.grpc.task_service_grpc_client import (
    TaskServiceGrpcClient,
    TaskState,
    task_service_grpc_client,
)
from app.llm.tools.task_tools.search_task.args import (
    AddQueryTaskArgs,
)


def _is_valid_task_id(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


class CreatePaperSearchTaskNode:
    def __init__(
        self,
        client: TaskServiceGrpcClient | None = None,
    ) -> None:
        self.client = client or task_service_grpc_client

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        existing_task_id = state.get("paper_service_task_id")

        if _is_valid_task_id(existing_task_id):
            return {
                "stage": "generating_source_queries",
                "status": "running",
                "error": None,
            }

        try:
            request = AddQueryTaskArgs.model_validate(
                {
                    "prompt": state.get("original_prompt"),
                    "searchTag": state.get("search_tag"),
                    "queryUnderstanding": state.get(
                        "query_understanding"
                    ),
                }
            )
        except Exception as exc:
            return {
                "stage": "blocked",
                "status": "blocked",
                "error": f"创建检索任务参数无效：{exc}",
            }

        payload = request.model_dump(
            mode="json",
            exclude_none=False,
        )

        try:
            result = await self.client.add_query_task(payload)
        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "create_task_payload": payload,
                "error": f"创建检索任务失败：{exc}",
            }

        if result.get("ok") is not True:
            return {
                "stage": "failed",
                "status": "failed",
                "create_task_payload": payload,
                "error": str(
                    result.get("error")
                    or "paper-service 创建检索任务失败"
                ),
                "warnings": [
                    *state.get("warnings", []),
                    "远端检索任务未创建，后续检索流程未启动。",
                ],
            }

        task_result = result.get("result") or {}
        task_id = task_result.get("task_id")

        if not _is_valid_task_id(task_id):
            return {
                "stage": "failed",
                "status": "failed",
                "create_task_payload": payload,
                "error": "task-service gRPC 创建搜索任务失败",
            }

        try:
            status_result = await self.client.update_task_status(
                task_id=task_id,
                task_state=TaskState.SEARCH_RUNNING,
            )
        except Exception as exc:
            status_result = {
                "ok": False,
                "error": str(exc),
            }
        warnings = list(state.get("warnings", []))
        task_status_update_error = None
        if status_result.get("ok") is not True:
            task_status_update_error = str(
                status_result.get("error") or "unknown error"
            )
            warnings.append(
                "Task status update to SEARCH_RUNNING failed: "
                f"{task_status_update_error}"
            )

        return {
            "stage": "generating_source_queries",
            "status": "running",
            "create_task_payload": payload,
            "paper_service_task_id": task_id,
            "progress": {
                **state.get("progress", {}),
                "task_created": 1,
                "task_running_status_updated": (
                    1 if task_status_update_error is None else 0
                ),
            },
            "warnings": warnings,
            "task_status_update_error": task_status_update_error,
            "remote_task_state": (
                TaskState.SEARCH_RUNNING.name
                if task_status_update_error is None
                else None
            ),
            "error": None,
        }
