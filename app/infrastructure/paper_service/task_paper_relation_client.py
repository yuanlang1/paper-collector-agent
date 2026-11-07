from typing import Any

from app.infrastructure.paper_service.http_client import (
    CLIENT_ERRORS,
    ServiceClient,
    list_value,
    object_value,
    positive_id,
)


class TaskPaperRelationClient(ServiceClient):
    @staticmethod
    def _validate_task_id(task_id: int) -> None:
        positive_id(task_id)
        if task_id > 2147483647:
            raise ValueError("task_id must be between 1 and 2147483647")

    async def get_unindexed_paper_rag_infos_by_task_id(
        self, task_id: int
    ) -> dict[str, Any]:
        try:
            self._validate_task_id(task_id)
            response = await self.http.request(
                "GET",
                "/internal/task-paper-relation/unindexed-rag-infos",
                params={"taskId": task_id},
            )
            data = object_value(response["data"])
            infos = [object_value(item) for item in list_value(data["paperInfos"])]
            return self._success(
                {
                    "task_id": data["taskId"],
                    "paper_infos": [
                        {
                            "paper_id": positive_id(item["paperId"], "paper_id"),
                            "title": item.get("title") or "",
                            "url": item.get("url") or "",
                            "paper_abstract": item.get("paperAbstract") or "",
                        }
                        for item in infos
                    ],
                },
                response,
            )
        except CLIENT_ERRORS as exc:
            return self._error(exc)

    async def get_paper_ids_by_task_id(self, task_id: int) -> dict[str, Any]:
        try:
            self._validate_task_id(task_id)
            response = await self.http.request(
                "GET",
                "/internal/task-paper-relation/paper-ids",
                params={"taskId": task_id},
            )
            return self._success(
                {
                    "task_id": task_id,
                    "paper_ids": [
                        positive_id(value, "paper_id")
                        for value in list_value(response["data"])
                    ],
                },
                response,
            )
        except CLIENT_ERRORS as exc:
            return self._error(exc)


task_paper_relation_client = TaskPaperRelationClient()
