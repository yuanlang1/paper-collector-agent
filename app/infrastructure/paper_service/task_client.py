from typing import Any

from app.infrastructure.paper_service.http_client import (
    CLIENT_ERRORS,
    ServiceClient,
    positive_id,
)
from app.infrastructure.paper_service.types import TaskState
from app.llm.graph.workflows.paper_search_schemas import PaperTypeCode


class TaskServiceClient(ServiceClient):
    async def add_query_task(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            tag = request["searchTag"]
            response = await self.http.request(
                "POST",
                "/internal/task/add-task",
                json={
                    "prompt": request["prompt"],
                    "searchTag": {
                        "yearTag": tag["yearTag"],
                        "paperTag": [
                            PaperTypeCode(code).api_value for code in tag["paperTag"]
                        ],
                        "sourceTag": [
                            getattr(source, "value", source)
                            for source in tag["sourceTag"]
                        ],
                    },
                    "promptUnderstanding": request["queryUnderstanding"],
                },
            )
            return self._success(
                {"task_id": positive_id(response["data"]), "status": "task_created",},
                response,
            )
        except CLIENT_ERRORS as exc:
            return self._error(exc)

    async def update_task_status(
        self,
        *,
        task_id: int,
        task_state: TaskState | int,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        try:
            positive_id(task_id)
            if isinstance(task_state, bool):
                raise ValueError("unsupported task_state")
            state = TaskState(task_state)
            response = await self.http.request(
                "POST",
                "/internal/task/update-status",
                json={
                    "taskId": task_id,
                    "taskState": int(state),
                    "errorMessage": str(error_message or "")
                    if state == TaskState.SEARCH_FAILED
                    else None,
                },
            )
            if response["data"] is not True:
                raise ValueError("Task status update was not confirmed")
            return self._success({"updated": True}, response)
        except CLIENT_ERRORS as exc:
            return self._error(exc)

    async def batch_save_task_paper_relations(
        self, relations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            if not relations:
                raise ValueError("relations must not be empty")
            items = []
            for relation in relations:
                star = relation.get("star", relation.get("recommendation_stars"))
                if isinstance(star, bool):
                    raise ValueError("star must be an integer")
                items.append(
                    {
                        "taskId": positive_id(relation["task_id"]),
                        "paperId": positive_id(relation["paper_id"], "paper_id"),
                        "star": int(star) if star is not None else None,
                        "recommendationMarkdown": str(
                            relation.get("recommendation_markdown") or ""
                        ),
                        "reasons": str(
                            relation.get("reasons")
                            or relation.get("recommendation_reason")
                            or ""
                        ),
                    }
                )
            response = await self.http.request(
                "POST",
                "/internal/task-paper-relation/batch-save",
                json={"relations": items},
            )
            if type(response["data"]) is not int or response["data"] < 0:
                raise ValueError("Invalid saved relation count")
            return self._success({"saved_count": response["data"]}, response)
        except CLIENT_ERRORS as exc:
            return self._error(exc)


task_service_client = TaskServiceClient()
