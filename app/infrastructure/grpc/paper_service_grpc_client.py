from datetime import date, datetime, time, timezone
from typing import Any

import grpc
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.wrappers_pb2 import Int32Value

from app.config import settings
from app.infrastructure.grpc.grpc_channel_pool import paper_service_grpc_channel_pool
from app.protos.paper.v1 import paper_pb2, paper_pb2_grpc


class PaperServiceGrpcClient:
    async def _get_stub(
        self,
    ) -> paper_pb2_grpc.PaperInternalServiceStub:
        channel = await paper_service_grpc_channel_pool.get_channel()
        return paper_pb2_grpc.PaperInternalServiceStub(channel)

    @staticmethod
    def _authors_text(value: Any) -> str:
        if isinstance(value, list):
            return ", ".join(
                str(author).strip()
                for author in value
                if str(author).strip()
            )

        return str(value or "").strip()

    @staticmethod
    def _validate_task_id(task_id: int) -> None:
        if (
            not isinstance(task_id, int)
            or isinstance(task_id, bool)
            or task_id <= 0
        ):
            raise ValueError("task_id must be a positive integer")

    @staticmethod
    def _timestamp_to_iso8601(value: Timestamp) -> str:
        return (
            value.ToDatetime(tzinfo=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    @staticmethod
    def _timestamp(value: Any) -> Timestamp | None:
        if value is None or value == "":
            return None

        if isinstance(value, datetime):
            resolved = value
        elif isinstance(value, date):
            resolved = datetime.combine(value, time.min)
        elif isinstance(value, str):
            resolved = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            raise ValueError("published_date must be an ISO-8601 date or datetime")

        if resolved.tzinfo is None:
            resolved = resolved.replace(tzinfo=timezone.utc)

        timestamp = Timestamp()
        timestamp.FromDatetime(resolved.astimezone(timezone.utc))
        return timestamp

    @classmethod
    def _build_request(
        cls,
        item: dict[str, Any],
    ) -> paper_pb2.SavePaperRequest:
        client_key = str(item["client_key"]).strip()
        paper_info = item["paper_info"]

        if not client_key:
            raise ValueError("client_key must not be blank")
        if not isinstance(paper_info, dict):
            raise ValueError("paper_info must be an object")

        title = str(paper_info.get("title") or "").strip()
        authors = cls._authors_text(paper_info.get("authors"))
        venue_id = paper_info.get("venue_id")

        if not title or not authors:
            raise ValueError("title and authors must not be blank")
        if (
            not isinstance(venue_id, int)
            or isinstance(venue_id, bool)
            or venue_id < 0
        ):
            raise ValueError("venue_id must be a non-negative integer")

        request = paper_pb2.SavePaperRequest(
            client_key=client_key,
            title=title,
            authors=authors,
            paper_abstract=str(
                paper_info.get("paper_abstract")
                or paper_info.get("abstract")
                or ""
            ),
            ai_abstract=str(paper_info.get("ai_abstract") or ""),
            doi=str(paper_info.get("doi") or ""),
            venue_id=venue_id,
            keywords=str(paper_info.get("keywords") or ""),
            source=str(paper_info.get("source") or ""),
            pdf_url=str(paper_info.get("pdf_url") or ""),
            abstract_url=str(paper_info.get("abstract_url") or ""),
            oss_name=str(paper_info.get("oss_name") or ""),
        )

        published_date = cls._timestamp(paper_info.get("published_date"))
        if published_date is not None:
            request.published_date.CopyFrom(published_date)

        citations = paper_info.get("citations")
        if citations is not None:
            if isinstance(citations, bool):
                raise ValueError("citations must be an integer")
            request.citations.CopyFrom(Int32Value(value=int(citations)))

        return request

    async def batch_save_papers(
        self,
        papers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not papers:
            return {
                "ok": False,
                "result": None,
                "error": "papers must not be empty",
                "metadata": {"source": "paper_service_grpc"},
            }

        try:
            stub = await self._get_stub()
            response = await stub.BatchSavePapers(
                paper_pb2.BatchSavePapersRequest(
                    papers=[self._build_request(paper) for paper in papers]
                ),
                timeout=settings.PAPER_SERVICE_GRPC_TIMEOUT_SECONDS,
            )

            if not response.success:
                return {
                    "ok": False,
                    "result": None,
                    "error": response.message or "Batch save papers failed",
                    "metadata": {
                        "source": "paper_service_grpc",
                        "code": response.code,
                    },
                }

            return {
                "ok": True,
                "result": {
                    "papers": [
                        {
                            "client_key": item.client_key,
                            "paper_id": item.id,
                            "status": "saved",
                        }
                        for item in response.data
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

        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return {
                "ok": False,
                "result": None,
                "error": str(exc),
                "metadata": {"source": "paper_service_grpc"},
            }

    async def get_task_review_papers(
        self,
        task_id: int,
    ) -> dict[str, Any]:
        try:
            self._validate_task_id(task_id)

            stub = await self._get_stub()
            response = await stub.GetTaskReviewPapers(
                paper_pb2.GetTaskReviewPapersRequest(task_id=task_id),
                timeout=settings.PAPER_SERVICE_GRPC_TIMEOUT_SECONDS,
            )

            if not response.success:
                return {
                    "ok": False,
                    "result": None,
                    "error": response.message or "Get task review papers failed",
                    "metadata": {
                        "source": "paper_service_grpc",
                        "code": response.code,
                    },
                }

            return {
                "ok": True,
                "result": {
                    "task_id": response.task_id,
                    "task_status": response.task_status,
                    "papers": [
                        {
                            "paper_id": item.paper_id,
                            "title": item.title,
                            "authors": [
                                author.strip()
                                for author in item.authors
                                if author.strip()
                            ],
                            "published_date": (
                                self._timestamp_to_iso8601(
                                    item.published_date
                                )
                                if item.HasField("published_date")
                                else None
                            ),
                            "doi": item.doi or None,
                            "rag_status": (item.rag_status or "PENDING").strip().lower(),
                            "chunk_count": item.chunk_count,
                        }
                        for item in response.papers
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


paper_service_grpc_client = PaperServiceGrpcClient()
