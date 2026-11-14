from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from app.llm.tools.venue_tools.easy_scholar import easy_scholar_venue_handler
from app.schemas.venue import VenueInfoDraft, VenueRecord
from app.services.venue_service import VenueService


def _text(value: Any) -> str | None:
    if value is None:
        return None

    normalized = str(value).strip()
    return normalized or None


def _decimal(value: Any) -> Decimal:
    normalized = _text(value)
    if normalized is None:
        return Decimal(0)

    try:
        result = Decimal(normalized)
    except InvalidOperation:
        return Decimal(0)

    return result if result.is_finite() else Decimal(0)


def _custom_rank_value(
    custom_ranks: Any,
    dataset_name: str,
) -> str | None:
    if not isinstance(custom_ranks, list):
        return None

    for custom_rank in custom_ranks:
        if not isinstance(custom_rank, Mapping):
            continue

        abb_name = _text(custom_rank.get("abbName"))
        current_rank = _text(custom_rank.get("currentRank"))

        if (
            abb_name is not None
            and abb_name.casefold() == dataset_name.casefold()
            and current_rank is not None
        ):
            return current_rank

    return None


def _crossref_paper_type_code(paper: Mapping[str, Any]) -> int:
    metadata = paper.get("crossref_metadata")
    if not isinstance(metadata, Mapping):
        return 0

    value = metadata.get("paper_type_code")
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else 0
    )


def _build_easy_scholar_venue(
    data: Mapping[str, Any],
) -> VenueInfoDraft:
    official_rank = data.get("official_rank")
    official_rank = official_rank if isinstance(official_rank, Mapping) else {}

    publication_name = _text(data.get("publication_name"))
    custom_ccf_rank = _custom_rank_value(
        data.get("custom_rank"),
        "CCF",
    )

    if not publication_name:
        raise ValueError("EasyScholar 未返回 publication_name。")

    return VenueInfoDraft(
        standard_name=publication_name,
        acronym=_text(data.get("abbreviation")),
        type=0,
        sci_rank=_text(official_rank.get("sci")),
        ccf_rank=custom_ccf_rank or _text(official_rank.get("ccf")),
        sci_if=_decimal(official_rank.get("sciif") or official_rank.get("sciIf")),
        sci_up=_text(official_rank.get("sciUp")),
        sci_up_small=_text(official_rank.get("sciUpSmall")),
        core_rank=_text(official_rank.get("pku")),
    )


class VenueResolver:
    def __init__(
        self,
        *,
        easy_scholar_handler: Any | None = None,
        venue_service: VenueService | None = None,
    ) -> None:
        self.venue_service = venue_service or VenueService()
        self.easy_scholar_handler = easy_scholar_handler or easy_scholar_venue_handler

    async def _query_easy_scholar(
        self,
        publication_name: str,
    ) -> dict[str, Any] | None:
        result = await self.easy_scholar_handler(
            {"publication_name": publication_name},
            None,
        )

        if result.get("ok") is not True:
            return None

        data = result.get("data")
        return data if isinstance(data, dict) else None

    async def resolve_venue(
        self,
        *,
        venue_candidates: list[str],
        draft_name: str | None = None,
    ) -> tuple[VenueInfoDraft | None, str]:
        candidates = dict.fromkeys(
            candidate.strip()
            for candidate in venue_candidates
            if isinstance(candidate, str) and 0 < len(candidate.strip()) <= 500
        )

        if not candidates:
            return None, "missing"

        for candidate in candidates:
            record, source = await self.venue_service.get_by_name(candidate)
            if record is not None:
                return record, source

        for candidate in candidates:
            tool_data = await self._query_easy_scholar(candidate)

            if tool_data is None:
                continue

            try:
                venue = _build_easy_scholar_venue(tool_data)
            except ValueError:
                continue

            record, source = await self.venue_service.get_by_name(venue.standard_name)
            return (record, source) if record is not None else (venue, "easy_scholar")

        if draft_name and draft_name in candidates:
            return VenueInfoDraft(standard_name=draft_name), "source_metadata"
        return None, "missing"

    async def resolve_paper(
        self,
        paper: dict[str, Any],
        venue_name: str | None = None,
        *,
        use_source_fallback: bool = True,
    ) -> tuple[str, str | None]:
        paper_info = paper["paper_info"]
        paper_info["venue_id"] = None
        title = paper_info["title"]
        try:
            crossref = paper.get("crossref_metadata") or {}
            draft_name = _text(venue_name)
            candidates = []
            if use_source_fallback and not draft_name:
                draft_name = _text(crossref.get("venue_title"))
                candidates = paper.get("venue_candidates", [])
            venue, source = await self.resolve_venue(
                venue_candidates=[draft_name, *candidates]
                if draft_name
                else candidates,
                draft_name=draft_name,
            )
            if venue is None:
                paper["venue_resolution"] = {
                    "status": "missing",
                    "source": source,
                    "venue_id": None,
                }
                return "missing", None
            paper_type_code = _crossref_paper_type_code(paper)
            if paper_type_code and not isinstance(venue, VenueRecord):
                venue = venue.model_copy(update={"type": paper_type_code})
            venue_id = venue.id if isinstance(venue, VenueRecord) else None
            paper_info["venue_id"] = venue_id
            paper["venue_resolution"] = {
                "status": "resolved",
                "source": source,
                "venue_id": venue_id,
                "venue_draft": venue.model_dump(mode="json"),
            }
            return "resolved", None
        except Exception as exc:
            paper["venue_resolution"] = {
                "status": "failed",
                "source": None,
                "venue_id": None,
            }
            return "failed", f"{title} 的 venue 解析失败：{exc}"
