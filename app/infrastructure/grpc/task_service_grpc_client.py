from enum import IntEnum
from typing import Any

import grpc
from google.protobuf.wrappers_pb2 import BoolValue, Int32Value

from app.config import settings
from app.infrastructure.grpc.grpc_channel_pool import paper_service_grpc_channel_pool
from app.protos.task.v1 import task_pb2, task_pb2_grpc


SOURCE_TYPE_CODES = {
    "arXiv": 1,
    "DBLP": 2,
    "Google Scholar": 3,
}


class TaskState(IntEnum):
    SEARCH_PENDING = 0
    SEARCH_RUNNING = 1
    SEARCH_COMPLETED = 2
    SEARCH_FAILED = 3
    SEARCH_PARTIAL_COMPLETED = 4
    CANCELLED = 5
    RAG_RUNNING = 6
    RAG_FAILED = 7
    RAG_COMPLETED = 8


class TaskServiceGrpcClient:
    async def _get_stub(
        self,
    ) -> task_pb2_grpc.TaskInternalServiceStub:
        channel = await paper_service_grpc_channel_pool.get_channel()

        return task_pb2_grpc.TaskInternalServiceStub(channel)

    @staticmethod
    def _source_type_code(source: Any) -> int:
        value = getattr(source, "value", source)

        if value not in SOURCE_TYPE_CODES:
            raise ValueError(f"不支持的检索来源：{value}")

        return SOURCE_TYPE_CODES[value]

    @staticmethod
    def _build_prompt_understanding(
        understanding: dict[str, Any],
    ) -> task_pb2.PromptUnderstanding:
        result = task_pb2.PromptUnderstanding(
            topic=understanding["topic"],
            subfields=understanding.get("subfields", []),
            intent=understanding["intent"],
            keywords=understanding.get("keywords", []),
            synonyms=understanding.get("synonyms", []),
            include_terms=understanding.get("includeTerms", []),
            exclude_terms=understanding.get("excludeTerms", []),
            reasoning=understanding["reasoning"],
        )

        year_from = understanding.get("yearFrom")
        if year_from is not None:
            result.year_from.CopyFrom(
                Int32Value(value=year_from)
            )

        year_to = understanding.get("yearTo")
        if year_to is not None:
            result.year_to.CopyFrom(
                Int32Value(value=year_to)
            )

        requires_code = understanding.get("requiresCode")
        if requires_code is not None:
            result.requires_code.CopyFrom(
                BoolValue(value=requires_code)
            )

        return result

    async def add_query_task(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            search_tag = request["searchTag"]
            understanding = request["queryUnderstanding"]

            stub = await self._get_stub()

            response = await stub.AddQueryTask(
                task_pb2.AddQueryTaskRequest(
                    prompt=request["prompt"],
                    search_tag=task_pb2.QueryTaskTag(
                        year_tag=search_tag["yearTag"],
                        paper_type_codes=[
                            int(code)
                            for code in search_tag["paperTag"]
                        ],
                        source_type_codes=[
                            self._source_type_code(source)
                            for source in search_tag["sourceTag"]
                        ],
                    ),
                    prompt_understanding=self._build_prompt_understanding(
                        understanding
                    ),
                ),
                timeout=settings.PAPER_SERVICE_GRPC_TIMEOUT_SECONDS,
            )

            if not response.success:
                return {
                    "ok": False,
                    "result": None,
                    "error": response.message or "创建检索任务失败",
                    "metadata": {
                        "source": "paper_service_grpc",
                        "code": response.code,
                    },
                }

            if not response.HasField("data"):
                return {
                    "ok": False,
                    "result": None,
                    "error": "paper-service 未返回任务 ID",
                    "metadata": {
                        "source": "paper_service_grpc",
                        "code": response.code,
                    },
                }

            return {
                "ok": True,
                "result": {
                    "task_id": response.data.value,
                    "status": "task_created",
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
                    f"gRPC 调用失败："
                    f"code={exc.code().name}, details={exc.details()}"
                ),
                "metadata": {
                    "source": "paper_service_grpc",
                },
            }

        except (KeyError, RuntimeError, ValueError) as exc:
            return {
                "ok": False,
                "result": None,
                "error": str(exc),
                "metadata": {
                    "source": "paper_service_grpc",
                },
            }

    async def update_task_status(
        self,
        *,
        task_id: int,
        task_state: TaskState | int,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        try:
            if (
                not isinstance(task_id, int)
                or isinstance(task_id, bool)
                or task_id <= 0
            ):
                raise ValueError("task_id must be a positive integer")

            if isinstance(task_state, bool):
                raise ValueError("unsupported task_state")

            try:
                state = TaskState(task_state)
            except ValueError as exc:
                raise ValueError("unsupported task_state") from exc

            stub = await self._get_stub()
            response = await stub.UpdateTaskStatus(
                task_pb2.UpdateTaskStatusRequest(
                    task_id=task_id,
                    task_state=int(state),
                    error_message=(
                        str(error_message or "")
                        if state == TaskState.SEARCH_FAILED
                        else ""
                    ),
                ),
                timeout=settings.PAPER_SERVICE_GRPC_TIMEOUT_SECONDS,
            )

            if not response.success or not response.updated:
                return {
                    "ok": False,
                    "result": {"updated": response.updated},
                    "error": response.message or "task status update failed",
                    "metadata": {
                        "source": "paper_service_grpc",
                        "code": response.code,
                    },
                }

            return {
                "ok": True,
                "result": {"updated": True},
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

    async def batch_save_task_paper_relations(
        self,
        relations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not relations:
            return {
                "ok": False,
                "result": None,
                "error": "relations must not be empty",
                "metadata": {"source": "paper_service_grpc"},
            }

        try:
            grpc_relations = []
            for relation in relations:
                task_id = relation["task_id"]
                paper_id = relation["paper_id"]

                if (
                    not isinstance(task_id, int)
                    or isinstance(task_id, bool)
                    or task_id <= 0
                ):
                    raise ValueError("task_id must be a positive integer")
                if (
                    not isinstance(paper_id, int)
                    or isinstance(paper_id, bool)
                    or paper_id <= 0
                ):
                    raise ValueError("paper_id must be a positive integer")

                item = task_pb2.TaskPaperRelationItem(
                    task_id=task_id,
                    paper_id=paper_id,
                    recommendation_markdown=str(
                        relation.get("recommendation_markdown") or ""
                    ),
                    reasons=str(
                        relation.get("reasons")
                        or relation.get("recommendation_reason")
                        or ""
                    ),
                )

                star = relation.get(
                    "star",
                    relation.get("recommendation_stars"),
                )
                if star is not None:
                    if isinstance(star, bool):
                        raise ValueError("star must be an integer")
                    item.star.CopyFrom(Int32Value(value=int(star)))

                grpc_relations.append(item)

            stub = await self._get_stub()
            response = await stub.BatchSaveTaskPaperRelations(
                task_pb2.BatchSaveTaskPaperRelationsRequest(
                    relations=grpc_relations,
                ),
                timeout=settings.PAPER_SERVICE_GRPC_TIMEOUT_SECONDS,
            )

            if not response.success:
                return {
                    "ok": False,
                    "result": None,
                    "error": (
                        response.message
                        or "Batch save task-paper relations failed"
                    ),
                    "metadata": {
                        "source": "paper_service_grpc",
                        "code": response.code,
                    },
                }

            return {
                "ok": True,
                "result": {"saved_count": response.data},
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

        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            return {
                "ok": False,
                "result": None,
                "error": str(exc),
                "metadata": {"source": "paper_service_grpc"},
            }


task_service_grpc_client = TaskServiceGrpcClient()
