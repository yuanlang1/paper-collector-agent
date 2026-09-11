from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Dict, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import settings
from app.llm.tools.registry import Tool, ToolExecutionContext
from app.llm.tools.search_tools.common import RETRYABLE_STATUS_CODES, clean_text, elapsed_ms, ensure_list, parse_year

DBLP_API_URL = settings.DBLP_API_URL


DblpPaperType = Literal[
    "all",
    "Journal Articles",
    "Conference and Workshop Papers",
    "Books and Theses",
    "Parts in Books or Collections",
    "Editorship",
    "Reference Works",
    "Data and Artifacts",
    "Informal and Other Publications",
]


class DblpSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=500)
    f: int = Field(0, ge=0)
    h: int = Field(5, ge=1, le=1000)
    total_limit: int = Field(5, ge=1)
    max_pages: int = Field(1, ge=1, le=20)
    year_from: int | None = Field(None, ge=1936, le=2100)
    year_to: int | None = Field(None, ge=1936, le=2100)
    venue: str | None = Field(None, min_length=1, max_length=200)
    paper_type: DblpPaperType = "all"

    @model_validator(mode="after")
    def validate_year_range(self) -> "DblpSearchArgs":
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from must be less than or equal to year_to")
        return self


class DblpUpstreamResponseError(Exception):
    code: str
    retryable: bool

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        content_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.content_type = content_type


class DblpResponseParseError(DblpUpstreamResponseError):
    """DBLP 返回了无法解析的 JSON 响应。"""

    code = "UPSTREAM_PARSE_ERROR"
    retryable = False


class DblpAccessChallengeError(DblpUpstreamResponseError):
    """DBLP 返回了访问挑战页，而不是 API 数据。"""

    code = "UPSTREAM_ACCESS_CHALLENGE"
    retryable = True


def _parse_author(author: Any) -> str:
    if isinstance(author, str):
        return clean_text(author)

    if isinstance(author, dict):
        return clean_text(
            author.get("text")
            or author.get("#text")
            or author.get("@name")
        )

    return ""


def _parse_authors(info: dict[str, Any]) -> list[str]:
    authors_object = info.get("authors")

    if isinstance(authors_object, dict):
        authors = authors_object.get("author")
    else:
        authors = authors_object

    return [
        name
        for name in (
            _parse_author(item)
            for item in ensure_list(authors)
        )
        if name
    ]


def _parse_ee_url(value: Any) -> str | None:
    for item in ensure_list(value):
        if isinstance(item, str):
            url = clean_text(item)
        elif isinstance(item, dict):
            url = clean_text(
                item.get("text") or item.get("#text") or item.get("@href"))
        else:
            url = ""

        if url:
            return url

    return None


def _normalize_dblp_paper(
    info: dict[str, Any],
) -> dict[str, Any] | None:
    title = clean_text(info.get("title"))
    if not title:
        return None

    year = parse_year(info.get("year"))
    doi = clean_text(info.get("doi")) or None
    dblp_key = clean_text(info.get("key")) or None
    dblp_url = clean_text(info.get("url")) or None
    external_url = _parse_ee_url(info.get("ee"))

    return {
        "source": "dblp",
        "source_id": (dblp_key or doi or dblp_url or title),
        "title": title,
        "authors": _parse_authors(info),
        "abstract": None,
        "published": None,
        "year": year,
        "updated": None,
        "venue": clean_text(info.get("venue")) or None,
        "publication_type": clean_text(info.get("type")) or None,
        "doi": doi,
        "dblp_key": dblp_key,
        "landing_url": dblp_url,
        "external_url": external_url,
        "pdf_url": None,
    }


def _passes_local_filters(
    paper: dict[str, Any],
    *,
    year_from: int | None,
    year_to: int | None,
    venue: str | None,
    paper_type: str,
) -> bool:
    year = paper.get("year")

    if year_from is not None:
        if year is None or year < year_from:
            return False

    if year_to is not None:
        if year is None or year > year_to:
            return False

    if venue:
        paper_venue = str(paper.get("venue") or "").casefold()
        if venue.casefold() not in paper_venue:
            return False

    if (paper_type != "all" and paper.get("publication_type") != paper_type):
        return False

    return True


def _is_dblp_access_challenge(response: httpx.Response) -> bool:
    content_type = response.headers.get("content-type", "").casefold()
    if "json" in content_type:
        return False

    body = response.text[:10_000].casefold()
    return (
        "making sure you're not a bot" in body
        or "within.website" in body
    )


async def _fetch_dblp(
    *,
    client: httpx.AsyncClient,
    request_params: dict[str, Any],
    retries: int = 2,
    retry_delay_seconds: float = 2.0,
) -> dict[str, Any]:
    retryable_status_codes = RETRYABLE_STATUS_CODES
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = await client.get(
                DBLP_API_URL,
                params=request_params,
                headers={
                    "User-Agent": "paper-collector-agent/0.1",
                },
            )

            if (
                response.status_code in retryable_status_codes
                and attempt < retries
            ):
                await asyncio.sleep(retry_delay_seconds)
                continue

            response.raise_for_status()

            content_type = response.headers.get("content-type")
            if _is_dblp_access_challenge(response):
                raise DblpAccessChallengeError(
                    "DBLP returned an access challenge page.",
                    status_code=response.status_code,
                    content_type=content_type,
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise DblpResponseParseError(
                    "DBLP returned invalid JSON",
                    status_code=response.status_code,
                    content_type=content_type,
                ) from exc

            if not isinstance(payload, dict):
                raise DblpResponseParseError(
                    "DBLP response must be a JSON object",
                    status_code=response.status_code,
                    content_type=content_type,
                )

            return payload

        except httpx.RequestError as exc:
            last_error = exc

            if attempt < retries:
                await asyncio.sleep(retry_delay_seconds)
                continue

            raise

    if last_error:
        raise last_error

    raise RuntimeError("DBLP request failed")


def _parse_dblp_result(
    data: dict[str, Any],
) -> dict[str, Any]:
    hits_object = data.get("result", {}).get("hits", {})

    hits = hits_object.get("hit", [])
    papers: list[dict[str, Any]] = []

    for hit in ensure_list(hits):
        if not isinstance(hit, dict):
            continue

        info = hit.get("info")
        if not isinstance(info, dict):
            continue

        paper = _normalize_dblp_paper(info)
        if paper:
            papers.append(paper)

    return {
        "total_results": parse_year(hits_object.get("@total")) or 0,
        "papers": papers,
    }


async def dblp_search_handler(
    params: Dict[str, Any],
    context: ToolExecutionContext,
) -> Dict[str, Any]:
    del context

    args = DblpSearchArgs.model_validate(params)
    started_at = time.perf_counter()
    query = clean_text(args.query)

    def failure(
        *,
        code: str,
        message: str,
        retryable: bool,
        status_code: int | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "duration_ms": elapsed_ms(started_at),
        }
        if status_code is not None:
            metadata["status_code"] = status_code
        if content_type:
            metadata["content_type"] = content_type

        return {
            "ok": False,
            "source": "dblp",
            "query": query,
            "returned_count": 0,
            "papers": [],
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
            },
            "metadata": metadata,
        }

    try:
        if not query:
            raise ValueError("query cannot be empty")

        papers: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        current_offset = args.f
        pages_fetched = 0
        raw_fetched_count = 0
        total_results = 0
        stop_reason = "upstream_exhausted"
        page_warning: str | None = None

        async with httpx.AsyncClient(timeout=30.0) as client:
            while (
                len(papers) < args.total_limit
                and pages_fetched < args.max_pages
            ):
                page_size = min(
                    args.h,
                    args.total_limit - len(papers),
                )
                try:
                    response_data = await _fetch_dblp(
                        client=client,
                        request_params={
                            "q": query,
                            "format": "json",
                            "f": current_offset,
                            "h": page_size,
                        },
                    )
                    parsed = _parse_dblp_result(response_data)
                except (
                    httpx.HTTPStatusError,
                    httpx.RequestError,
                    DblpUpstreamResponseError,
                ) as exc:
                    if not papers:
                        raise
                    error_code = getattr(exc, "code", None)
                    page_warning = (
                        f"第 {pages_fetched + 1} 页请求失败："
                        f"{f'[{error_code}] ' if error_code else ''}{exc}"
                    )
                    stop_reason = "upstream_error"
                    break

                page_papers = parsed["papers"]
                raw_page_count = len(page_papers)
                pages_fetched += 1
                raw_fetched_count += raw_page_count
                total_results = parsed["total_results"]
                new_paper_count = 0
                filtered_page_count = 0

                for paper in page_papers:
                    if not _passes_local_filters(
                        paper,
                        year_from=args.year_from,
                        year_to=args.year_to,
                        venue=args.venue,
                        paper_type=args.paper_type,
                    ):
                        continue
                    filtered_page_count += 1

                    source_id = str(paper.get("source_id") or "")
                    if not source_id or source_id in seen_ids:
                        continue
                    seen_ids.add(source_id)
                    papers.append(paper)
                    new_paper_count += 1
                    if len(papers) >= args.total_limit:
                        break

                next_offset = current_offset + raw_page_count
                if len(papers) >= args.total_limit:
                    stop_reason = "total_limit_reached"
                    break
                if pages_fetched >= args.max_pages:
                    stop_reason = "max_pages_reached"
                    break
                if not raw_page_count or next_offset >= total_results:
                    stop_reason = "upstream_exhausted"
                    break
                if raw_page_count < page_size:
                    stop_reason = "upstream_exhausted"
                    break
                if filtered_page_count and not new_paper_count:
                    stop_reason = "no_new_results"
                    break

                current_offset = next_offset

        next_offset = current_offset + raw_page_count if pages_fetched else None
        has_next = (
            stop_reason in {"total_limit_reached", "max_pages_reached"}
            and next_offset is not None
            and next_offset < total_results
        )

        return {
            "ok": True,
            "source": "dblp",
            "query": query,
            "returned_count": len(papers),
            "papers": papers,
            "error": None,
            "status": "partial_failed" if page_warning else "success",
            "warnings": [page_warning] if page_warning else [],
            "metadata": {
                "total_results": total_results,
                "candidate_count": raw_fetched_count,
                "pagination": {
                    "initial_offset": args.f,
                    "page_size": args.h,
                    "total_limit": args.total_limit,
                    "max_pages": args.max_pages,
                    "pages_fetched": pages_fetched,
                    "raw_fetched_count": raw_fetched_count,
                    "returned_count": len(papers),
                    "has_next": has_next,
                    "next_offset": next_offset if has_next else None,
                    "stop_reason": stop_reason,
                },
                "filters": {
                    "year_from": args.year_from,
                    "year_to": args.year_to,
                    "venue": args.venue,
                    "paper_type": args.paper_type,
                },
                "duration_ms": elapsed_ms(started_at),
            },
        }

    except httpx.HTTPStatusError as exc:
        return failure(
            code="UPSTREAM_HTTP_ERROR",
            message=exc.response.text[:500],
            retryable=exc.response.status_code in RETRYABLE_STATUS_CODES,
            status_code=exc.response.status_code,
        )

    except httpx.RequestError as exc:
        return failure(
            code="UPSTREAM_REQUEST_ERROR",
            message=str(exc),
            retryable=True,
        )

    except DblpUpstreamResponseError as exc:
        return failure(
            code=exc.code,
            message=str(exc),
            retryable=exc.retryable,
            status_code=exc.status_code,
            content_type=exc.content_type,
        )

    except ValueError as exc:
        return failure(
            code="INVALID_ARGUMENT",
            message=str(exc),
            retryable=False,
        )

    except Exception as exc:
        return failure(
            code="INTERNAL_ERROR",
            message=str(exc),
            retryable=False,
        )


DBLP_SEARCH_TOOL = Tool(
    name="dblp_search",
    description=(
        "从 DBLP 少量检索计算机领域出版物并直接返回论文元数据。"
        "适用于论文标题、作者、研究主题、会议或期刊查询。"
        "单次默认返回 5 篇，最多 20 篇。"
        "DBLP 通常不提供摘要和 PDF，因此这些字段可能为空。"
        "不用于批量检索、分页、文件导出或跨来源去重。"
    ),
    input_schema=DblpSearchArgs.model_json_schema(),
    fn=dblp_search_handler,
    requires_confirmation=False,
)
