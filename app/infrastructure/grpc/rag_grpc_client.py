from typing import Any
from uuid import UUID

import grpc

from app.config import settings
from app.infrastructure.grpc.grpc_channel_pool import (
    paper_service_grpc_channel_pool,
)
from app.protos.rag.v1 import rag_pb2, rag_pb2_grpc


_RESULT_STATUS_CODES = {
    "ready": rag_pb2.RAG_BATCH_RESULT_READY,
    "skipped": rag_pb2.RAG_BATCH_RESULT_SKIPPED,
    "failed": rag_pb2.RAG_BATCH_RESULT_FAILED,
}


class RagGrpcClient:
    async def _get_stub(self) -> rag_pb2_grpc.RagInternalServiceStub:
        channel = await paper_service_grpc_channel_pool.get_channel()
        return rag_pb2_grpc.RagInternalServiceStub(channel)

    @staticmethod
    def _validate_task_id(task_id: int) -> None:
        if (
            not isinstance(task_id, int)
            or isinstance(task_id, bool)
            or task_id <= 0
        ):
            raise ValueError("task_id must be a positive integer")

    @staticmethod
    def _validate_batch_id(batch_id: str) -> str:
        if not isinstance(batch_id, str):
            raise ValueError("batch_id must be a UUID")

        value = batch_id.strip()
        try:
            UUID(value)
        except ValueError as exc:
            raise ValueError("batch_id must be a UUID") from exc

        return value

    @classmethod
    def _build_result(cls, item: dict[str, Any]) -> rag_pb2.RagPaperResult:
        paper_id = item["paper_id"]
        status = item["status"]
        chunk_count = item.get("chunk_count", 0)
        error = str(item.get("error") or "").strip()

        if (
            not isinstance(paper_id, int)
            or isinstance(paper_id, bool)
            or paper_id <= 0
        ):
            raise ValueError("paper_id must be a positive integer")
        if status not in _RESULT_STATUS_CODES:
            raise ValueError("status must be ready, skipped, or failed")
        if isinstance(chunk_count, bool) or not isinstance(chunk_count, int):
            raise ValueError("chunk_count must be an integer")
        if status == "ready" and chunk_count < 0:
            raise ValueError("ready result requires a non-negative chunk_count")
        if status in {"skipped", "failed"} and not error:
            raise ValueError(f"{status} result requires an error")

        return rag_pb2.RagPaperResult(
            paper_id = paper_id,
            status = _RESULT_STATUS_CODES[status],
            chunk_count = chunk_count,
            error = error,
        )

    async def claim_task_rag_papers(
        self,
        *,
        task_id: int,
        batch_id: str,
        limit: int,
        lease_seconds: int,
    ) -> dict[str, Any]:
        try:
            self._validate_task_id(task_id)
            batch_id = self._validate_batch_id(batch_id)
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
                raise ValueError("limit must be an integer between 1 and 100")
            if (
                isinstance(lease_seconds, bool)
                or not isinstance(lease_seconds, int)
                or not 60 <= lease_seconds <= 3600
            ):
                raise ValueError(
                    "lease_seconds must be an integer between 60 and 3600"
                )

            stub = await self._get_stub()
            response = await stub.ClaimTaskRagPapers(
                rag_pb2.ClaimTaskRagPapersRequest(
                    task_id = task_id,
                    batch_id = batch_id,
                    limit = limit,
                    lease_seconds = lease_seconds,
                ),
                timeout = settings.PAPER_SERVICE_GRPC_TIMEOUT_SECONDS,
            )

            if not response.success:
                return self._service_error(
                    response.code,
                    response.message or "Claim task RAG papers failed",
                )

            return {
                "ok": True,
                "result": {
                    "task_id": response.task_id,
                    "batch_id": response.batch_id,
                    "papers": [
                        {
                            "paper_id": paper.paper_id,
                            "title": paper.title,
                            "url": paper.url,
                            "paper_abstract": paper.paper_abstract,
                            "oss_name": paper.oss_name,
                        }
                        for paper in response.papers
                    ],
                    "claimed_count": response.claimed_count,
                    "task_state": response.task_state,
                },
                "error": None,
                "metadata": self._metadata(response.code),
            }

        except grpc.aio.AioRpcError as exc:
            return self._rpc_error(exc)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            return self._input_error(exc)

    async def complete_task_rag_batch(
        self,
        *,
        task_id: int,
        batch_id: str,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            self._validate_task_id(task_id)
            batch_id = self._validate_batch_id(batch_id)
            if not results:
                raise ValueError("results must not be empty")

            grpc_results = [self._build_result(item) for item in results]
            paper_ids = [result.paper_id for result in grpc_results]
            if len(paper_ids) != len(set(paper_ids)):
                raise ValueError("paper_id must be unique in one RAG batch")

            stub = await self._get_stub()
            response = await stub.CompleteTaskRagBatch(
                rag_pb2.CompleteTaskRagBatchRequest(
                    task_id = task_id,
                    batch_id = batch_id,
                    results = grpc_results,
                ),
                timeout = settings.PAPER_SERVICE_GRPC_TIMEOUT_SECONDS,
            )

            if not response.success:
                return self._service_error(
                    response.code,
                    response.message or "Complete task RAG batch failed",
                )

            return {
                "ok": True,
                "result": {
                    "task_id": response.task_id,
                    "batch_id": response.batch_id,
                    "task_state": response.task_state,
                    "total_count": response.total_count,
                    "pending_count": response.pending_count,
                    "indexing_count": response.indexing_count,
                    "ready_count": response.ready_count,
                    "skipped_count": response.skipped_count,
                    "failed_count": response.failed_count,
                    "progress_percent": response.progress_percent,
                },
                "error": None,
                "metadata": self._metadata(response.code),
            }

        except grpc.aio.AioRpcError as exc:
            return self._rpc_error(exc)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            return self._input_error(exc)

    @staticmethod
    def _metadata(code: int | None = None) -> dict[str, Any]:
        metadata: dict[str, Any] = {"source": "paper_service_grpc"}
        if code is not None:
            metadata["code"] = code
        return metadata

    @classmethod
    def _service_error(cls, code: int, error: str) -> dict[str, Any]:
        return {
            "ok": False,
            "result": None,
            "error": error,
            "metadata": cls._metadata(code),
        }

    @classmethod
    def _rpc_error(cls, exc: grpc.aio.AioRpcError) -> dict[str, Any]:
        return {
            "ok": False,
            "result": None,
            "error": (
                f"gRPC call failed: "
                f"code = {exc.code().name}, details = {exc.details()}"
            ),
            "metadata": cls._metadata(),
        }

    @classmethod
    def _input_error(cls, exc: Exception) -> dict[str, Any]:
        return {
            "ok": False,
            "result": None,
            "error": str(exc),
            "metadata": cls._metadata(),
        }


rag_grpc_client = RagGrpcClient()
