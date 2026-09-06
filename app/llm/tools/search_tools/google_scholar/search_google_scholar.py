from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.config import settings
from app.llm.tools.registry import Tool
from app.llm.tools.search_tools.common import RETRYABLE_STATUS_CODES, clean_text, elapsed_ms
from app.llm.tools.search_tools.google_scholar.search_args import (
    GoogleScholarSearchArgs,
)

SERPAPI_SEARCH_URL = settings.SERPAPI_SEARCH_URL


class SerpApiResponseParseError(Exception):
    """SerpApi 响应无法解析为有效 JSON。"""


class SerpApiSearchError(Exception):
    """SerpApi 返回了业务错误。"""


class SerpApiConfigurationError(Exception):
    """SerpApi 配置缺失或无效。"""


def _extract_year(value: str | None) -> int | None:
    if not value:
        return None

    match = re.search(r"\b(19|20)\d{2}\b", value)
    return int(match.group(0)) if match else None


def _extract_authors(
    publication_info: dict[str, Any],
) -> list[str]:
    authors = publication_info.get("authors") or []

    result: list[str] = []

    for author in authors:
        if isinstance(author, dict):
            name = clean_text(author.get("name"))
        else:
            name = clean_text(author)

        if name:
            result.append(name)

    return result


def _extract_pdf_url(
    item: dict[str, Any],
) -> str | None:
    for resource in item.get("resources") or []:
        if not isinstance(resource, dict):
            continue

        link = resource.get("link")
        file_format = clean_text(resource.get("file_format")).upper()

        if isinstance(link, str) and file_format == "PDF":
            return link

    link = item.get("link")
    if isinstance(link, str) and link.lower().endswith(".pdf"):
        return link

    return None


def _extract_cited_by(
    item: dict[str, Any],
) -> dict[str, Any]:
    inline_links = item.get("inline_links") or {}
    cited_by = inline_links.get("cited_by") or {}

    total = cited_by.get("total")
    try:
        citation_count = int(total) if total is not None else None
    except (TypeError, ValueError):
        citation_count = None

    return {
        "citation_count": citation_count,
        "cites_id": (
            str(cited_by["cites_id"])
            if cited_by.get("cites_id")
            else None
        ),
        "scholar_cited_by_url": cited_by.get("link"),
    }


def _extract_versions(
    item: dict[str, Any],
) -> dict[str, Any]:
    inline_links = item.get("inline_links") or {}
    versions = inline_links.get("versions") or {}

    total = versions.get("total")
    try:
        versions_count = int(total) if total is not None else None
    except (TypeError, ValueError):
        versions_count = None

    return {
        "cluster_id": (
            str(versions["cluster_id"])
            if versions.get("cluster_id")
            else None
        ),
        "versions_count": versions_count,
    }


def _parse_google_scholar_item(
    item: dict[str, Any],
) -> dict[str, Any] | None:
    title = clean_text(item.get("title"))
    if not title:
        return None

    publication_info = item.get("publication_info") or {}
    if not isinstance(publication_info, dict):
        publication_info = {}

    publication_summary = clean_text(publication_info.get("summary"))
    snippet = clean_text(item.get("snippet")) or None
    result_id = clean_text(item.get("result_id")) or None
    landing_url = item.get("link")
    cited_by = _extract_cited_by(item)
    versions = _extract_versions(item)

    return {
        "source": "google_scholar",
        "source_id": (
            result_id
            or cited_by["cites_id"]
            or landing_url
            or title
        ),
        "title": title,
        "authors": _extract_authors(publication_info),
        "abstract": None,
        "snippet": snippet,
        "published": None,
        "year": _extract_year(publication_summary) or _extract_year(snippet),
        "updated": None,
        "venue": None,
        "publication_summary": publication_summary or None,
        "doi": None,
        "citation_count": cited_by["citation_count"],
        "scholar_result_id": result_id,
        "cites_id": cited_by["cites_id"],
        "cluster_id": versions["cluster_id"],
        "versions_count": versions["versions_count"],
        "landing_url": landing_url,
        "pdf_url": _extract_pdf_url(item),
        "scholar_cited_by_url": cited_by[
            "scholar_cited_by_url"
        ],
    }


async def _request_serpapi(
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
                SERPAPI_SEARCH_URL,
                params=request_params,
            )

            if (
                response.status_code in retryable_status_codes
                and attempt < retries
            ):
                await asyncio.sleep(retry_delay_seconds)
                continue

            response.raise_for_status()

            try:
                return response.json()
            except ValueError as exc:
                raise SerpApiResponseParseError(
                    "SerpApi returned invalid JSON"
                ) from exc

        except httpx.RequestError as exc:
            last_error = exc

            if attempt < retries:
                await asyncio.sleep(retry_delay_seconds)
                continue

            raise

    if last_error:
        raise last_error

    raise RuntimeError("SerpApi request failed")


def _build_serpapi_params(
    args: GoogleScholarSearchArgs,
) -> dict[str, Any]:
    if not settings.SERPAPI_API_KEY:
        raise SerpApiConfigurationError(
            "SERPAPI_API_KEY is not configured"
        )

    params: dict[str, Any] = {
        "engine": "google_scholar",
        "q": args.query,
        "api_key": settings.SERPAPI_API_KEY,
        "output": "json",
        "hl": "en",
        "start": args.start,
        "num": args.num,
        # 默认排除专利。
        "as_sdt": "0",
        # 默认排除 citation records，只保留普通学术结果。
        "as_vis": "1",
        # 保留 Google Scholar 默认的相似结果/省略结果过滤。
        "filter": "1",
    }

    if args.year_from is not None:
        params["as_ylo"] = args.year_from

    if args.year_to is not None:
        params["as_yhi"] = args.year_to

    if args.review_only:
        params["as_rr"] = "1"

    return params


def _validate_serpapi_result(
    raw_result: dict[str, Any],
) -> None:
    if raw_result.get("error"):
        raise SerpApiSearchError(
            clean_text(raw_result["error"])
        )

    metadata = raw_result.get("search_metadata") or {}
    status = clean_text(metadata.get("status"))

    if status and status.lower() != "success":
        raise SerpApiSearchError(
            f"SerpApi search status: {status}"
        )


async def google_scholar_search_handler(
    params: dict[str, Any],
    _db: Session,
) -> dict[str, Any]:
    del _db

    args = GoogleScholarSearchArgs.model_validate(params)
    query = clean_text(args.query)
    started_at = time.perf_counter()

    def failure(
        *,
        code: str,
        message: str,
        retryable: bool,
        status_code: int | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "duration_ms": elapsed_ms(started_at),
        }

        if status_code is not None:
            metadata["status_code"] = status_code

        return {
            "ok": False,
            "source": "google_scholar",
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
        current_start = args.start
        pages_fetched = 0
        raw_fetched_count = 0
        total_results: Any = None
        search_id: Any = None
        stop_reason = "upstream_exhausted"
        page_warning: str | None = None

        async with httpx.AsyncClient(timeout=30.0) as client:
            while (
                len(papers) < args.total_limit
                and pages_fetched < args.max_pages
            ):
                page_size = min(
                    args.num,
                    args.total_limit - len(papers),
                )
                page_args = args.model_copy(
                    update={"start": current_start, "num": page_size}
                )
                request_params = _build_serpapi_params(page_args)

                try:
                    raw_result = await _request_serpapi(
                        client=client,
                        request_params=request_params,
                    )
                    _validate_serpapi_result(raw_result)
                except (
                    httpx.HTTPStatusError,
                    httpx.RequestError,
                    SerpApiResponseParseError,
                    SerpApiSearchError,
                ) as exc:
                    if not papers:
                        raise
                    page_warning = f"第 {pages_fetched + 1} 页请求失败：{exc}"
                    stop_reason = "upstream_error"
                    break

                page_papers = [
                    paper
                    for paper in (
                        _parse_google_scholar_item(item)
                        for item in (
                            raw_result.get("organic_results") or []
                        )
                        if isinstance(item, dict)
                    )
                    if paper is not None
                ]
                raw_page_count = len(page_papers)
                pages_fetched += 1
                raw_fetched_count += raw_page_count
                search_information = raw_result.get("search_information") or {}
                search_metadata = raw_result.get("search_metadata") or {}
                serpapi_pagination = raw_result.get("serpapi_pagination") or {}
                total_results = search_information.get("total_results")
                search_id = search_metadata.get("id")
                has_upstream_next = bool(
                    isinstance(serpapi_pagination, dict)
                    and (
                        serpapi_pagination.get("next")
                        or serpapi_pagination.get("next_link")
                    )
                )
                new_paper_count = 0

                for paper in page_papers:
                    source_id = str(paper.get("source_id") or "")
                    if not source_id or source_id in seen_ids:
                        continue
                    seen_ids.add(source_id)
                    papers.append(paper)
                    new_paper_count += 1
                    if len(papers) >= args.total_limit:
                        break

                next_start = current_start + page_size
                if len(papers) >= args.total_limit:
                    stop_reason = "total_limit_reached"
                    break
                if pages_fetched >= args.max_pages:
                    stop_reason = "max_pages_reached"
                    break
                if not has_upstream_next or not raw_page_count:
                    stop_reason = "upstream_exhausted"
                    break
                if not new_paper_count:
                    stop_reason = "no_new_results"
                    break

                current_start = next_start

        next_offset = current_start + page_size if pages_fetched else None
        has_next = (
            stop_reason in {"total_limit_reached", "max_pages_reached"}
            and next_offset is not None
            and has_upstream_next
        )

        return {
            "ok": True,
            "source": "google_scholar",
            "query": query,
            "returned_count": len(papers),
            "papers": papers,
            "error": None,
            "status": "partial_failed" if page_warning else "success",
            "warnings": [page_warning] if page_warning else [],
            "metadata": {
                "total_results": total_results,
                "serpapi_search_id": search_id,
                "pagination": {
                    "initial_offset": args.start,
                    "page_size": args.num,
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
                    "review_only": args.review_only,
                    "exclude_patents": True,
                    "exclude_citations": True,
                },
                "duration_ms": elapsed_ms(started_at),
            },
        }

    except httpx.HTTPStatusError as exc:
        return failure(
            code="SERPAPI_HTTP_ERROR",
            message=exc.response.text[:500],
            retryable=exc.response.status_code in RETRYABLE_STATUS_CODES,
            status_code=exc.response.status_code,
        )

    except httpx.RequestError as exc:
        return failure(
            code="SERPAPI_REQUEST_ERROR",
            message=str(exc),
            retryable=True,
        )

    except SerpApiConfigurationError as exc:
        return failure(
            code="SERPAPI_CONFIGURATION_ERROR",
            message=str(exc),
            retryable=False,
        )

    except SerpApiResponseParseError as exc:
        return failure(
            code="SERPAPI_RESPONSE_INVALID",
            message=str(exc),
            retryable=False,
        )

    except SerpApiSearchError as exc:
        return failure(
            code="SERPAPI_SEARCH_ERROR",
            message=str(exc),
            retryable=False,
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


GOOGLE_SCHOLAR_SEARCH_TOOL = Tool(
    name="google_scholar_search",
    description=(
        "通过 SerpApi 的 Google Scholar 接口少量检索学术论文。"
        "适用于补充跨领域候选论文、获取引用量、版本数及 PDF 候选链接。"
        "单次默认返回 5 篇，最多 10 篇。"
        "Google Scholar 返回的是摘要片段 snippet，而非完整论文摘要。"
        "不用于批量检索、分页、文件导出或跨来源补全。"
    ),
    input_schema=GoogleScholarSearchArgs.model_json_schema(),
    fn=google_scholar_search_handler,
    requires_confirmation=False,
)
