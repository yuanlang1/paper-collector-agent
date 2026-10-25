from typing import Any
from uuid import UUID

from app.infrastructure.paper_service.http_client import (
    CLIENT_ERRORS,
    ServiceClient,
    list_value,
    object_value,
    positive_id,
)
from app.infrastructure.paper_service.types import TaskState


class RagClient(ServiceClient):
    @staticmethod
    def _validate_batch_id(batch_id: str) -> str:
        if not isinstance(batch_id, str):
            raise ValueError("batch_id must be a UUID")
        value = batch_id.strip()
        UUID(value)
        return value

    @staticmethod
    def _build_result(item: dict[str, Any]) -> dict[str, Any]:
        paper_id = positive_id(item["paper_id"], "paper_id")
        status = item["status"]
        chunk_count = item.get("chunk_count", 0)
        error = str(item.get("error") or "").strip()
        if status not in {"ready", "skipped", "failed"}:
            raise ValueError("status must be ready, skipped, or failed")
        if type(chunk_count) is not int:
            raise ValueError("chunk_count must be an integer")
        if status == "ready" and chunk_count < 0:
            raise ValueError("ready result requires a non-negative chunk_count")
        if status in {"skipped", "failed"} and not error:
            raise ValueError(f"{status} result requires an error")
        return {
            "paperId": paper_id,
            "status": status.upper(),
            "chunkCount": chunk_count,
            "error": error,
        }

    async def claim_task_rag_papers(
        self, *, task_id: int, batch_id: str, limit: int, lease_seconds: int,
    ) -> dict[str, Any]:
        try:
            positive_id(task_id)
            batch_id = self._validate_batch_id(batch_id)
            if type(limit) is not int or not 1 <= limit <= 100:
                raise ValueError("limit must be an integer between 1 and 100")
            if type(lease_seconds) is not int or not 60 <= lease_seconds <= 3600:
                raise ValueError("lease_seconds must be an integer between 60 and 3600")
            response = await self.http.request(
                "POST",
                "/internal/rag/claim",
                json={
                    "taskId": task_id,
                    "batchId": batch_id,
                    "limit": limit,
                    "leaseSeconds": lease_seconds,
                },
            )
            data = object_value(response["data"])
            papers = [object_value(item) for item in list_value(data["papers"])]
            return self._success(
                {
                    "task_id": data["taskId"],
                    "batch_id": data["batchId"],
                    "papers": [
                        {
                            "paper_id": positive_id(item["paperId"], "paper_id"),
                            "title": item.get("title") or "",
                            "url": item.get("url") or "",
                            "paper_abstract": item.get("paperAbstract") or "",
                            "oss_name": item.get("ossName") or "",
                        }
                        for item in papers
                    ],
                    "claimed_count": data["claimedCount"],
                    "task_state": TaskState(data["taskState"]).name,
                },
                response,
            )
        except CLIENT_ERRORS as exc:
            return self._error(exc)

    async def complete_task_rag_batch(
        self, *, task_id: int, batch_id: str, results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            positive_id(task_id)
            batch_id = self._validate_batch_id(batch_id)
            if not results:
                raise ValueError("results must not be empty")
            items = [self._build_result(item) for item in results]
            if len({item["paperId"] for item in items}) != len(items):
                raise ValueError("paper_id must be unique in one RAG batch")
            response = await self.http.request(
                "POST",
                "/internal/rag/complete",
                json={"taskId": task_id, "batchId": batch_id, "results": items,},
            )
            data = object_value(response["data"])
            return self._success(
                {
                    "task_id": data["taskId"],
                    "batch_id": data["batchId"],
                    "task_state": TaskState(data["taskState"]).name,
                    "total_count": data["totalCount"],
                    "pending_count": data["pendingCount"],
                    "indexing_count": data["indexingCount"],
                    "ready_count": data["readyCount"],
                    "skipped_count": data["skippedCount"],
                    "failed_count": data["failedCount"],
                    "progress_percent": data["progressPercent"],
                },
                response,
            )
        except CLIENT_ERRORS as exc:
            return self._error(exc)


rag_client = RagClient()
