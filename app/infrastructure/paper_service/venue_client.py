from decimal import Decimal
from typing import Any

from app.infrastructure.paper_service.http_client import (
    CLIENT_ERRORS,
    ServiceClient,
    list_value,
    object_value,
    positive_id,
)


class VenueServiceClient(ServiceClient):
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
    ) -> dict[str, Any]:
        if not standard_name.strip():
            raise ValueError("standard_name must not be blank")
        return {
            "standardName": standard_name.strip(),
            "type": int(type) if type is not None else None,
            "acronym": acronym,
            "sciRank": sci_rank,
            "ccfRank": ccf_rank,
            "sciIf": float(sci_if) if sci_if is not None else None,
            "sciUp": sci_up,
            "sciUpSmall": sci_up_small,
            "coreRank": core_rank,
        }

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
            response = await self.http.request(
                "POST",
                "/internal/venue/save",
                json=self._build_save_request(
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
            )
            return self._success(
                {"id": positive_id(response["data"], "venue_id")}, response
            )
        except CLIENT_ERRORS as exc:
            return self._error(exc)

    async def batch_save_venues(self, venues: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            if not venues:
                raise ValueError("venues must not be empty")
            requests = [
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
                for venue in venues
            ]
            response = await self.http.request(
                "POST", "/internal/venue/batch-save", json={"venues": requests}
            )
            saved = [object_value(item) for item in list_value(response["data"])]
            return self._success(
                {
                    "venues": [
                        {
                            "standard_name": item["standardName"],
                            "id": positive_id(item["id"], "venue_id"),
                        }
                        for item in saved
                    ]
                },
                response,
            )
        except CLIENT_ERRORS as exc:
            return self._error(exc)


venue_service_client = VenueServiceClient()
