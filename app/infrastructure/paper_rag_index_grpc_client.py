from __future__ import annotations

from typing import Any

import grpc

from app.config import settings
from app.infrastructure.grpc_channel_pool import paper_service_grpc_channel_pool
from app.protos.rag.v1 import paper_rag_index_pb2, paper_rag_index_pb2_grpc


class PaperRagIndexGrpcClient:
    async def _get_stub(
        self,
    ) -> paper_rag_index_pb2_grpc.PaperRagIndexInternalServiceStub:
        channel = await paper_service_grpc_channel_pool.get_channel()
        return paper_rag_index_pb2_grpc.PaperRagIndexInternalServiceStub(channel)

    @staticmethod
    def _items(items: list[dict[str, Any]]) -> list[Any]:
        requests = []
        for item in items:
            paper_id = item.get("paper_id")
            content_hash = str(item.get("content_hash") or "").strip()
            if not isinstance(paper_id, int) or isinstance(paper_id, bool) or paper_id <= 0:
                raise ValueError("paper_id must be a positive integer")
            if not content_hash:
                raise ValueError("content_hash must not be blank")
            requests.append(
                paper_rag_index_pb2.PaperHashItem(
                    paper_id=paper_id,
                    content_hash=content_hash,
                )
            )
        if not requests:
            raise ValueError("items must not be empty")
        return requests

    @staticmethod
    def _view(index: Any) -> dict[str, Any]:
        return {
            "paper_id": index.paper_id,
            "content_hash": index.content_hash,
            "index_status": index.index_status,
            "lightrag_document_id": index.lightrag_document_id or None,
            "lightrag_track_id": index.lightrag_track_id or None,
            "error_message": index.error_message or None,
        }

    async def ensure(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            stub = await self._get_stub()
            response = await stub.EnsurePaperRagIndexes(
                paper_rag_index_pb2.EnsurePaperRagIndexesRequest(items=self._items(items)),
                timeout=settings.PAPER_SERVICE_GRPC_TIMEOUT_SECONDS,
            )
            if not response.success:
                return self._failure(response.message, response.code)
            return self._success(
                {
                    "indexes": [
                        {"index": self._view(item.index), "should_index": item.should_index}
                        for item in response.indexes
                    ]
                },
                response.code,
            )
        except (grpc.aio.AioRpcError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            return self._exception_failure(exc)

    async def claim(
        self,
        items: list[dict[str, Any]],
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> dict[str, Any]:
        if not worker_id.strip():
            return self._local_failure("worker_id must not be blank")
        if lease_seconds <= 0:
            return self._local_failure("lease_seconds must be positive")
        try:
            stub = await self._get_stub()
            response = await stub.ClaimPaperRagIndexes(
                paper_rag_index_pb2.ClaimPaperRagIndexesRequest(
                    items=self._items(items),
                    worker_id=worker_id,
                    lease_seconds=lease_seconds,
                ),
                timeout=settings.PAPER_SERVICE_GRPC_TIMEOUT_SECONDS,
            )
            if not response.success:
                return self._failure(response.message, response.code)
            return self._success(
                {
                    "claimed": [
                        {
                            "paper_id": item.paper_id,
                            "lease_token": item.lease_token,
                            "index_status": item.index_status,
                            "lease_expires_at": item.lease_expires_at.ToJsonString(),
                        }
                        for item in response.claimed
                    ]
                },
                response.code,
            )
        except (grpc.aio.AioRpcError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            return self._exception_failure(exc)

    async def complete(
        self,
        *,
        paper_id: int,
        lease_token: str,
        content_hash: str,
        index_status: str,
        lightrag_track_id: str = "",
        lightrag_document_id: str = "",
        error_message: str = "",
    ) -> dict[str, Any]:
        if index_status not in {"ready", "failed"}:
            return self._local_failure("complete only accepts ready or failed")
        try:
            stub = await self._get_stub()
            response = await stub.CompletePaperRagIndex(
                paper_rag_index_pb2.CompletePaperRagIndexRequest(
                    paper_id=paper_id,
                    lease_token=lease_token,
                    content_hash=content_hash,
                    index_status=index_status,
                    lightrag_track_id=lightrag_track_id,
                    lightrag_document_id=lightrag_document_id,
                    error_message=error_message,
                ),
                timeout=settings.PAPER_SERVICE_GRPC_TIMEOUT_SECONDS,
            )
            if not response.success:
                return self._failure(response.message, response.code)
            return self._success({"accepted": response.accepted}, response.code)
        except grpc.aio.AioRpcError as exc:
            return self._exception_failure(exc)

    async def get(self, paper_ids: list[int]) -> dict[str, Any]:
        if not paper_ids or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in paper_ids):
            return self._local_failure("paper_ids must contain positive integers")
        try:
            stub = await self._get_stub()
            response = await stub.GetPaperRagIndexes(
                paper_rag_index_pb2.GetPaperRagIndexesRequest(paper_ids=paper_ids),
                timeout=settings.PAPER_SERVICE_GRPC_TIMEOUT_SECONDS,
            )
            if not response.success:
                return self._failure(response.message, response.code)
            return self._success({"indexes": [self._view(item) for item in response.indexes]}, response.code)
        except grpc.aio.AioRpcError as exc:
            return self._exception_failure(exc)

    async def skip(self, paper_ids: list[int], *, reason: str) -> dict[str, Any]:
        if not reason.strip():
            return self._local_failure("reason must not be blank")
        if not paper_ids or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in paper_ids):
            return self._local_failure("paper_ids must contain positive integers")
        try:
            stub = await self._get_stub()
            response = await stub.SkipPaperRagIndexes(
                paper_rag_index_pb2.SkipPaperRagIndexesRequest(paper_ids=paper_ids, reason=reason),
                timeout=settings.PAPER_SERVICE_GRPC_TIMEOUT_SECONDS,
            )
            if not response.success:
                return self._failure(response.message, response.code)
            return self._success({"skipped_count": response.skipped_count}, response.code)
        except grpc.aio.AioRpcError as exc:
            return self._exception_failure(exc)

    @staticmethod
    def _success(result: dict[str, Any], code: int) -> dict[str, Any]:
        return {"ok": True, "result": result, "error": None, "metadata": {"source": "paper_rag_index_grpc", "code": code}}

    @staticmethod
    def _failure(message: str, code: int) -> dict[str, Any]:
        return {"ok": False, "result": None, "error": message or "paper RAG index RPC failed", "metadata": {"source": "paper_rag_index_grpc", "code": code}}

    @staticmethod
    def _local_failure(message: str) -> dict[str, Any]:
        return {"ok": False, "result": None, "error": message, "metadata": {"source": "paper_rag_index_grpc"}}

    def _exception_failure(self, exc: Exception) -> dict[str, Any]:
        if isinstance(exc, grpc.aio.AioRpcError):
            message = f"gRPC call failed: code={exc.code().name}, details={exc.details()}"
        else:
            message = str(exc)
        return self._local_failure(message)


paper_rag_index_grpc_client = PaperRagIndexGrpcClient()
