from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field


class SearchSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_size: int = Field(default=5, strict=True, ge=1)


class OffsetCursor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    offset: int = Field(strict=True, ge=0)


@dataclass(frozen=True)
class SearchBatchResult:
    provider_id: str
    papers: list[dict[str, Any]]
    next_cursor: dict[str, Any] | None
    status: Literal["success", "partial", "failed"]
    warnings: list[str]
    error: str | None
    pages_fetched: int
    raw_fetched_count: int
    total_results: int | None
    stop_reason: str


class SearchAdapter(Protocol):
    @property
    def provider_id(self) -> str: ...

    async def search(
        self,
        *,
        query_args: Mapping[str, Any],
        settings: Mapping[str, Any],
        credentials: Mapping[str, str],
        cursor: Mapping[str, Any] | None,
        limit: int,
        max_pages: int,
        http_client: httpx.AsyncClient | None = None,
    ) -> SearchBatchResult: ...


def source_arguments(
    query_args: Mapping[str, Any],
    settings: Mapping[str, Any],
    cursor: Mapping[str, Any] | None,
    limit: int,
    max_pages: int,
    *,
    page_field: str,
    offset_field: str,
    page_max: int,
) -> dict[str, Any]:
    forbidden = {"start", "f", "num", "h", "max_results", "total_limit", "max_pages"}
    if forbidden.intersection(query_args):
        raise ValueError("query_args must not contain pagination fields")
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    if type(max_pages) is not int or not 1 <= max_pages <= 20:
        raise ValueError("max_pages must be between 1 and 20")
    config = SearchSettings.model_validate(dict(settings))
    if config.page_size > page_max:
        raise ValueError(f"page_size must not exceed {page_max}")
    offset = (
        OffsetCursor.model_validate(dict(cursor)).offset if cursor is not None else 0
    )
    return {
        **query_args,
        page_field: config.page_size,
        offset_field: offset,
        "total_limit": limit,
        "max_pages": max_pages,
    }


def batch_result(provider_id: str, result: dict[str, Any]) -> SearchBatchResult:
    metadata = result["metadata"]
    pagination = metadata["pagination"]
    total = metadata.get("total_results")
    return SearchBatchResult(
        provider_id=provider_id,
        papers=result["papers"],
        next_cursor={"offset": pagination["next_offset"]}
        if pagination["has_next"]
        else None,
        status="partial" if result["warnings"] else "success",
        warnings=list(result["warnings"]),
        error=None,
        pages_fetched=pagination["pages_fetched"],
        raw_fetched_count=pagination["raw_fetched_count"],
        total_results=int(total) if total is not None else None,
        stop_reason=pagination["stop_reason"],
    )


def failed_result(provider_id: str, error: str) -> SearchBatchResult:
    return SearchBatchResult(
        provider_id=provider_id,
        papers=[],
        next_cursor=None,
        status="failed",
        warnings=[],
        error=error,
        pages_fetched=0,
        raw_fetched_count=0,
        total_results=None,
        stop_reason="upstream_error",
    )
