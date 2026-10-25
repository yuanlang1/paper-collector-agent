from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.infrastructure.paper_service.task_client import (
    TaskServiceClient,
    task_service_client,
)
from app.infrastructure.paper_service.types import TaskState
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.tools.task_tools.search_task.search_task_create import AddQueryTaskArgs


def _is_valid_task_id(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


class CreatePaperSearchTaskNode:
    def __init__(
        self,
        client: TaskServiceClient | None = None,
        artifact_store: LocalArtifactStore | None = None,
    ) -> None:
        self.client = client or task_service_client
        self.artifact_store = artifact_store or LocalArtifactStore()

    async def _bind_recommendation_manifest(
        self,
        state: Mapping[str, Any],
        task_id: int,
    ) -> str | None:
        bound_artifact_uri = state.get(
            "task_bound_recommendation_manifest_artifact_ref"
        )
        if (
            isinstance(bound_artifact_uri, str)
            and bound_artifact_uri.startswith("artifact://")
        ):
            return bound_artifact_uri

        artifact_uri = state.get("recommendation_manifest_artifact_ref")
        if artifact_uri is None:
            return None
        child_run_id = state["child_run_id"]
        manifest = await self.artifact_store.read_json_uri(artifact_uri)
        papers = manifest["papers"]

        for paper in papers:
            relation_draft = paper["task_paper_relation_draft"]
            previous_task_id = relation_draft.get("search_task_id")
            if previous_task_id is not None and previous_task_id != task_id:
                raise ValueError("recommendation manifest is bound to another task")
            relation_draft["search_task_id"] = task_id

        manifest["papers"] = papers
        manifest["step_key"] = "bind_recommendations_to_task"
        manifest["search_task_id"] = task_id
        artifact = await self.artifact_store.write_json(
            run_id=child_run_id,
            step_key="bind_recommendations_to_task",
            source="task_service",
            kind="task_bound_recommendation_manifest_json",
            payload=manifest,
            count=len(papers),
            metadata={
                "input_manifest": artifact_uri,
                "search_task_id": task_id,
            },
        )
        return artifact.artifact_uri

    async def _create_remote_task(self, state: Mapping[str, Any]) -> dict[str, Any]:
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
                "error": "task-service HTTP 创建搜索任务失败",
            }

        return {"create_task_payload": payload, "paper_service_task_id": task_id}

    async def _mark_running(self, task_id: int, warnings: list[str]) -> dict[str, Any]:
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
            "stage": "persisting_papers",
            "status": "running",
            "paper_service_task_id": task_id,
            "progress": {
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

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        task_id = state.get("paper_service_task_id")
        created = not _is_valid_task_id(task_id)
        update = {}
        if created:
            update = await self._create_remote_task(state)
            if update.get("status") in {"failed", "blocked"}:
                return update
            task_id = update["paper_service_task_id"]
        try:
            bound_artifact_uri = await self._bind_recommendation_manifest(state, task_id)
        except Exception as exc:
            return {
                **update,
                "stage": "failed", "status": "failed",
                "paper_service_task_id": task_id,
                "error": f"绑定推荐论文与检索任务失败：{exc}",
            }
        if created:
            update.update(await self._mark_running(task_id, list(state.get("warnings", []))))
            update["progress"] = {**state.get("progress", {}), **update["progress"]}
        return {
            **update,
            "stage": "persisting_papers", "status": "running",
            "task_bound_recommendation_manifest_artifact_ref": bound_artifact_uri,
            "error": None,
        }
