from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from app.infrastructure.grpc.task_service_grpc_client import (
    TaskServiceGrpcClient,
    TaskState,
    task_service_grpc_client,
)
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
        client: TaskServiceGrpcClient | None = None,
        artifact_store: LocalArtifactStore | None = None,
    ) -> None:
        self.client = client or task_service_grpc_client
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
        if (
            not isinstance(artifact_uri, str)
            or not artifact_uri.startswith("artifact://")
        ):
            raise ValueError("missing recommendation manifest artifact")

        run_id = state.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("missing valid run_id")

        base_dir = self.artifact_store.base_dir.resolve()
        manifest_path = (
            base_dir / artifact_uri.removeprefix("artifact://")
        ).resolve()
        try:
            manifest_path.relative_to(base_dir)
        except ValueError as exc:
            raise ValueError(
                "recommendation manifest is outside artifact storage"
            ) from exc

        def read_manifest() -> dict[str, Any]:
            with manifest_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            if not isinstance(payload, dict):
                raise ValueError("recommendation manifest must be an object")
            return payload

        manifest = await asyncio.to_thread(read_manifest)
        papers = manifest.get("papers")
        if not isinstance(papers, list):
            raise ValueError("recommendation manifest papers must be a list")

        for paper in papers:
            if not isinstance(paper, dict):
                raise ValueError("recommended paper must be an object")
            relation_draft = paper.get("task_paper_relation_draft")
            if not isinstance(relation_draft, dict):
                raise ValueError("recommended paper is missing relation draft")
            previous_task_id = relation_draft.get("search_task_id")
            if previous_task_id is not None and previous_task_id != task_id:
                raise ValueError("recommendation manifest is bound to another task")
            relation_draft["search_task_id"] = task_id

        manifest["papers"] = papers
        manifest["step_key"] = "bind_recommendations_to_task"
        manifest["search_task_id"] = task_id
        artifact = await self.artifact_store.write_json(
            run_id=run_id,
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

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        existing_task_id = state.get("paper_service_task_id")

        if _is_valid_task_id(existing_task_id):
            try:
                bound_artifact_uri = await self._bind_recommendation_manifest(
                    state,
                    existing_task_id,
                )
            except Exception as exc:
                return {
                    "stage": "failed",
                    "status": "failed",
                    "paper_service_task_id": existing_task_id,
                    "error": f"绑定推荐论文与检索任务失败：{exc}",
                }
            return {
                "stage": "persisting_papers",
                "status": (
                    "partial_failed"
                    if state.get("degraded")
                    else "running"
                ),
                "task_bound_recommendation_manifest_artifact_ref": (
                    bound_artifact_uri
                ),
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
            bound_artifact_uri = await self._bind_recommendation_manifest(
                state,
                task_id,
            )
        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "create_task_payload": payload,
                "paper_service_task_id": task_id,
                "error": f"绑定推荐论文与检索任务失败：{exc}",
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
            "stage": "persisting_papers",
            "status": (
                "partial_failed"
                if state.get("degraded")
                else "running"
            ),
            "create_task_payload": payload,
            "paper_service_task_id": task_id,
            "task_bound_recommendation_manifest_artifact_ref": (
                bound_artifact_uri
            ),
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
