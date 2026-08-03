from typing import Any

import grpc

from app.config import settings
from app.infrastructure.grpc.grpc_channel_pool import paper_service_grpc_channel_pool
from app.protos.task_paper_relation.v1 import (
    task_paper_relation_pb2,
    task_paper_relation_pb2_grpc,
)


class TaskPaperRelationGrpcClient:
    async def _get_stub(
        self,
    ) -> task_paper_relation_pb2_grpc.TaskPaperRelationInternalServiceStub:
        channel = await paper_service_grpc_channel_pool.get_channel()
        return task_paper_relation_pb2_grpc.TaskPaperRelationInternalServiceStub(
            channel
        )

    @staticmethod
    def _validate_task_id(task_id: int) -> None:
        if (
            not isinstance(task_id, int)
            or isinstance(task_id, bool)
            or task_id <= 0
        ):
            raise ValueError("task_id must be a positive integer")

    async def get_unindexed_paper_rag_infos_by_task_id(
        self,
        task_id: int,
    ) -> dict[str, Any]:
        try:
            self._validate_task_id(task_id)
            stub = await self._get_stub()
            response = await stub.GetUnindexedPaperRagInfosByTaskId(
                task_paper_relation_pb2.TaskIdRequest(task_id=task_id),
                timeout=settings.PAPER_SERVICE_GRPC_TIMEOUT_SECONDS,
            )

            if not response.success:
                return {
                    "ok": False,
                    "result": None,
                    "error": (
                        response.message
                        or "Get unindexed paper RAG infos failed"
                    ),
                    "metadata": {
                        "source": "paper_service_grpc",
                        "code": response.code,
                    },
                }

            return {
                "ok": True,
                "result": {
                    "task_id": response.task_id,
                    "paper_infos": [
                        {
                            "paper_id": item.paper_id,
                            "title": item.title,
                            "url": item.url,
                            "paper_abstract": item.paper_abstract,
                        }
                        for item in response.paper_infos
                    ],
                },
                "error": None,
                "metadata": {
                    "source": "paper_service_grpc",
                    "code": response.code,
                },
            }

        except grpc.aio.AioRpcError as exc:
            return {
                "ok": False,
                "result": None,
                "error": (
                    f"gRPC call failed: "
                    f"code={exc.code().name}, details={exc.details()}"
                ),
                "metadata": {"source": "paper_service_grpc"},
            }

        except (RuntimeError, TypeError, ValueError) as exc:
            return {
                "ok": False,
                "result": None,
                "error": str(exc),
                "metadata": {"source": "paper_service_grpc"},
            }

    async def get_paper_ids_by_task_id(
        self,
        task_id: int,
    ) -> dict[str, Any]:
        try:
            self._validate_task_id(task_id)
            stub = await self._get_stub()
            response = await stub.GetPaperIdsByTaskId(
                task_paper_relation_pb2.TaskIdRequest(task_id=task_id),
                timeout=settings.PAPER_SERVICE_GRPC_TIMEOUT_SECONDS,
            )

            if not response.success:
                return {
                    "ok": False,
                    "result": None,
                    "error": response.message or "Get paper IDs failed",
                    "metadata": {
                        "source": "paper_service_grpc",
                        "code": response.code,
                    },
                }

            return {
                "ok": True,
                "result": {
                    "task_id": task_id,
                    "paper_ids": list(response.paper_ids),
                },
                "error": None,
                "metadata": {
                    "source": "paper_service_grpc",
                    "code": response.code,
                },
            }

        except grpc.aio.AioRpcError as exc:
            return {
                "ok": False,
                "result": None,
                "error": (
                    f"gRPC call failed: "
                    f"code={exc.code().name}, details={exc.details()}"
                ),
                "metadata": {"source": "paper_service_grpc"},
            }

        except (RuntimeError, TypeError, ValueError) as exc:
            return {
                "ok": False,
                "result": None,
                "error": str(exc),
                "metadata": {"source": "paper_service_grpc"},
            }


task_paper_relation_grpc_client = TaskPaperRelationGrpcClient()
