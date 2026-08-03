from decimal import Decimal
from typing import Any

import grpc

from app.config import settings
from app.infrastructure.grpc.grpc_channel_pool import paper_service_grpc_channel_pool
from app.protos.venue.v1 import venue_pb2, venue_pb2_grpc


class VenueServiceGrpcClient:
    @staticmethod
    def _build_save_request(
        *,
        standard_name: str,
        type: int | None = None,
        acronym: str = "",
        sci_rank: str = "",
        ccf_rank: str = "",
        sci_if: float | Decimal | None = None,
        sci_up: str = "",
        sci_up_small: str = "",
        core_rank: str = "",
    ) -> venue_pb2.SaveVenueRequest:
        request = venue_pb2.SaveVenueRequest(
            standard_name=standard_name,
            acronym=acronym,
            sci_rank=sci_rank,
            ccf_rank=ccf_rank,
            sci_up=sci_up,
            sci_up_small=sci_up_small,
            core_rank=core_rank,
        )
        if type is not None:
            request.type = int(type)
        if sci_if is not None:
            request.sci_if = float(sci_if)
        return request

    async def _get_stub(
        self,
    ) -> venue_pb2_grpc.VenueInternalServiceStub:
        channel = await paper_service_grpc_channel_pool.get_channel()

        return venue_pb2_grpc.VenueInternalServiceStub(channel)

    async def save_venue(
        self,
        *,
        standard_name: str,
        type: int | None = None,
        acronym: str = "",
        sci_rank: str = "",
        ccf_rank: str = "",
        sci_if: float | Decimal | None = None,
        sci_up: str = "",
        sci_up_small: str = "",
        core_rank: str = "",
    ) -> dict[str, Any]:
        try:
            stub = await self._get_stub()

            response = await stub.SaveVenue(
                self._build_save_request(
                    standard_name=standard_name,
                    type=type,
                    acronym=acronym,
                    sci_rank=sci_rank,
                    ccf_rank=ccf_rank,
                    sci_if=sci_if,
                    sci_up=sci_up,
                    sci_up_small=sci_up_small,
                    core_rank=core_rank,
                ),
                timeout=settings.PAPER_SERVICE_GRPC_TIMEOUT_SECONDS,
            )

            if not response.success:
                return {
                    "ok": False,
                    "result": None,
                    "error": response.message or "保存 Venue 失败",
                    "metadata": {
                        "source": "paper_service_grpc",
                        "code": response.code,
                    },
                }

            if not response.HasField("data"):
                return {
                    "ok": False,
                    "result": None,
                    "error": "paper-service 未返回 Venue ID",
                    "metadata": {
                        "source": "paper_service_grpc",
                        "code": response.code,
                    },
                }

            return {
                "ok": True,
                "result": {"id": response.data.value},
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

        except RuntimeError as exc:
            return {
                "ok": False,
                "result": None,
                "error": str(exc),
                "metadata": {
                    "source": "paper_service_grpc",
                },
            }

    async def batch_save_venues(
        self,
        venues: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not venues:
            return {
                "ok": False,
                "result": None,
                "error": "venues must not be empty",
                "metadata": {
                    "source": "paper_service_grpc",
                },
            }

        try:
            requests = []
            for venue in venues:
                requests.append(
                    self._build_save_request(
                        standard_name=str(venue["standard_name"]).strip(),
                        type=venue.get("type"),
                        acronym=str(venue.get("acronym") or ""),
                        sci_rank=str(venue.get("sci_rank") or ""),
                        ccf_rank=str(venue.get("ccf_rank") or ""),
                        sci_if=venue.get("sci_if"),
                        sci_up=str(venue.get("sci_up") or ""),
                        sci_up_small=str(venue.get("sci_up_small") or ""),
                        core_rank=str(venue.get("core_rank") or ""),
                    )
                )

            stub = await self._get_stub()
            response = await stub.BatchSaveVenues(
                venue_pb2.BatchSaveVenuesRequest(venues=requests),
                timeout=settings.PAPER_SERVICE_GRPC_TIMEOUT_SECONDS,
            )

            if not response.success:
                return {
                    "ok": False,
                    "result": None,
                    "error": response.message or "Batch save venues failed",
                    "metadata": {
                        "source": "paper_service_grpc",
                        "code": response.code,
                    },
                }

            return {
                "ok": True,
                "result": {
                    "venues": [
                        {
                            "standard_name": item.standard_name,
                            "id": item.id,
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
                "metadata": {
                    "source": "paper_service_grpc",
                },
            }

        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return {
                "ok": False,
                "result": None,
                "error": str(exc),
                "metadata": {
                    "source": "paper_service_grpc",
                },
            }


venue_service_grpc_client = VenueServiceGrpcClient()
