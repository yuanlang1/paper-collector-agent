from __future__ import annotations

import asyncio
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict

import httpx
from sqlalchemy.orm import Session

from app.config import settings
from app.llm.tools.base import AppToolSpec
from app.llm.tools.search_tools.arxiv.search_args import ArxivSearchArgs
from app.llm.tools.search_tools.common import RETRYABLE_STATUS_CODES, clean_text, elapsed_ms

ARXIV_API_URL = settings.ARXIV_API_URL

ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    "arxiv": "http://arxiv.org/schemas/atom",
}

ARXIV_CATEGORY_PATTERN = re.compile(
    r"^[a-zA-Z-]+(?:\.[a-zA-Z-]+)?$"
)


def _quote_if_needed(value: str) -> str:
    value = value.strip()
    if not value:
        return value

    if " " in value:
        return f'"{value}"'

    return value


def _build_category_clause(
    category: str | None,
) -> str | None:
    if not category:
        return None

    categories = [
        item.strip()
        for item in category.split(",")
        if item.strip()
    ]

    if not categories:
        return None

    invalid_categories = [
        item
        for item in categories
        if not ARXIV_CATEGORY_PATTERN.fullmatch(item)
    ]
    if invalid_categories:
        raise ValueError(f"invalid arXiv category: {invalid_categories[0]}")

    if len(categories) == 1:
        return f"cat:{categories[0]}"

    return "(" + " OR ".join(f"cat:{item}" for item in categories) + ")"


def _build_date_clause(
    year_from: int | None,
    year_to: int | None,
) -> str | None:
    if year_from is None and year_to is None:
        return None

    start = f"{year_from or 1991}01010000"
    end = f"{year_to or 9999}12312359"

    return f"submittedDate:[{start} TO {end}]"


def _build_search_query(
    args: ArxivSearchArgs,
) -> str:
    if not args.query:
        raise ValueError("query cannot be empty")

    field_by_type = {
        "topic": "all",
        "title": "ti",
        "author": "au",
        "abstract": "abs",
        "category": "cat",
    }

    field = field_by_type[args.search_type]
    query = args.query.strip()

    if args.search_type == "category":
        clauses = [f"cat:{query}"]
    else:
        clauses = [f"{field}:{_quote_if_needed(query)}"]

        category_clause = _build_category_clause(args.category)
        if category_clause:
            clauses.append(category_clause)

    date_clause = _build_date_clause(
        year_from=args.year_from,
        year_to=args.year_to,
    )
    if date_clause:
        clauses.append(date_clause)

    return " AND ".join(clauses)


def _build_request_params(
    args: ArxivSearchArgs,
) -> tuple[dict[str, Any], str | None]:
    sort_mapping = {
        "relevance": ("relevance", "descending"),
        "newest": ("submittedDate", "descending"),
        "updated": ("lastUpdatedDate", "descending"),
    }
    sort_by, sort_order = sort_mapping[args.sort]

    request_params: dict[str, Any] = {
        "start": args.start,
        "max_results": args.max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }

    if args.arxiv_ids:
        request_params["id_list"] = ",".join(
            dict.fromkeys(item.strip() for item in args.arxiv_ids if item.strip())
        )
        return request_params, None

    search_query = _build_search_query(args)
    request_params["search_query"] = search_query

    return request_params, search_query


def _extract_arxiv_id(arxiv_id_url: str) -> str:
    if not arxiv_id_url:
        return ""

    return arxiv_id_url.rstrip("/").split("/")[-1]


def _extract_version(arxiv_id: str) -> str | None:
    match = re.search(r"v\d+$", arxiv_id)
    return match.group(0) if match else None


def _read_feed_int(
    root: ET.Element,
    path: str,
    default: int = 0,
) -> int:
    value = root.findtext(
        path,
        default=str(default),
        namespaces=ATOM_NS,
    )

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_arxiv_error(
    root: ET.Element,
) -> str | None:
    entry = root.find("atom:entry", ATOM_NS)
    if entry is None:
        return None

    entry_id = clean_text(
        entry.findtext(
            "atom:id",
            default="",
            namespaces=ATOM_NS,
        )
    )
    if "api/errors" not in entry_id:
        return None

    title = clean_text(
        entry.findtext(
            "atom:title",
            default="",
            namespaces=ATOM_NS,
        )
    )
    summary = clean_text(
        entry.findtext(
            "atom:summary",
            default="",
            namespaces=ATOM_NS,
        )
    )

    return summary or title or "Unknown arXiv API error"


def _parse_arxiv_feed(
    xml_text: str,
    *,
    include_abstract: bool,
) -> dict[str, Any]:
    root = ET.fromstring(xml_text)

    api_error = _extract_arxiv_error(root)
    if api_error:
        raise ValueError(api_error)

    total_results = _read_feed_int(root, "opensearch:totalResults")

    papers: list[dict[str, Any]] = []

    for entry in root.findall("atom:entry", ATOM_NS):
        arxiv_id_url = clean_text(
            entry.findtext(
                "atom:id",
                default="",
                namespaces=ATOM_NS,
            )
        )
        arxiv_id = _extract_arxiv_id(arxiv_id_url)

        title = clean_text(
            entry.findtext(
                "atom:title",
                default="",
                namespaces=ATOM_NS,
            )
        )

        abstract = entry.findtext(
            "atom:summary",
            default="",
            namespaces=ATOM_NS,
        )

        authors = [
            clean_text(
                author.findtext(
                    "atom:name",
                    default="",
                    namespaces=ATOM_NS,
                )
            )
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        authors = [author for author in authors if author]

        primary_category_element = entry.find("arxiv:primary_category", ATOM_NS,)
        primary_category = (
            primary_category_element.attrib.get("term")
            if primary_category_element is not None
            else None
        )

        categories = [
            item.attrib["term"]
            for item in entry.findall("atom:category", ATOM_NS)
            if item.attrib.get("term")
        ]

        abstract_url = arxiv_id_url
        pdf_url: str | None = None

        for link in entry.findall("atom:link", ATOM_NS):
            href = link.attrib.get("href")
            rel = link.attrib.get("rel")
            link_type = link.attrib.get("type")
            title_attr = link.attrib.get("title")

            if href and rel == "alternate":
                abstract_url = href

            if href and ( title_attr == "pdf" or link_type == "application/pdf"):
                pdf_url = href

        doi = clean_text(
            entry.findtext(
                "arxiv:doi",
                default="",
                namespaces=ATOM_NS,
            )
        )
        comment = clean_text(
            entry.findtext(
                "arxiv:comment",
                default="",
                namespaces=ATOM_NS,
            )
        )
        journal_ref = clean_text(
            entry.findtext(
                "arxiv:journal_ref",
                default="",
                namespaces=ATOM_NS,
            )
        )

        papers.append(
            {
                "source": "arxiv",
                "source_id": arxiv_id,
                "arxiv_id": arxiv_id,
                "version": _extract_version(arxiv_id),
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "published": clean_text(
                    entry.findtext(
                        "atom:published",
                        default="",
                        namespaces=ATOM_NS,
                    )
                ) or None,
                "updated": clean_text(
                    entry.findtext(
                        "atom:updated",
                        default="",
                        namespaces=ATOM_NS,
                    )
                ) or None,
                "primary_category": primary_category,
                "categories": categories,
                "doi": doi or None,
                "comment": comment or None,
                "journal_ref": journal_ref or None,
                "abstract_url": abstract_url or None,
                "pdf_url": pdf_url,
            }
        )

    return {
        "total_results": total_results,
        "papers": papers,
    }


async def _fetch_arxiv_page(
    *,
    client: httpx.AsyncClient,
    request_params: dict[str, Any],
    retries: int,
    retry_delay_seconds: float,
) -> str:
    retryable_status_codes = RETRYABLE_STATUS_CODES

    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = await client.get(
                ARXIV_API_URL,
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
            return response.text

        except httpx.RequestError as exc:
            last_error = exc

            if attempt < retries:
                await asyncio.sleep(retry_delay_seconds)
                continue

            raise

    if last_error:
        raise last_error

    raise RuntimeError("arXiv request failed")


async def arxiv_search_handler(
    params: Dict[str, Any],
    _db: Session,
) -> Dict[str, Any]:
    del _db

    args = ArxivSearchArgs.model_validate(params)
    started_at = time.perf_counter()
    search_query: str | None = None

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
            "source": "arxiv",
            "query": args.query,
            "arxiv_ids": args.arxiv_ids,
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
        papers: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        current_start = args.start
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
                    args.max_results,
                    args.total_limit - len(papers),
                )
                page_args = args.model_copy(
                    update={
                        "start": current_start,
                        "max_results": page_size,
                    }
                )
                request_params, search_query = _build_request_params(
                    page_args,
                )

                try:
                    xml_text = await _fetch_arxiv_page(
                        client=client,
                        request_params=request_params,
                        retries=2,
                        retry_delay_seconds=2.0,
                    )
                    parsed = _parse_arxiv_feed(
                        xml_text,
                        include_abstract=args.include_abstract,
                    )
                except (
                    httpx.HTTPStatusError,
                    httpx.RequestError,
                    ET.ParseError,
                ) as exc:
                    if not papers:
                        raise
                    page_warning = f"第 {pages_fetched + 1} 页请求失败：{exc}"
                    stop_reason = "upstream_error"
                    break

                page_papers = parsed["papers"]
                raw_page_count = len(page_papers)
                pages_fetched += 1
                raw_fetched_count += raw_page_count
                total_results = parsed["total_results"]

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

                next_start = current_start + raw_page_count
                if len(papers) >= args.total_limit:
                    stop_reason = "total_limit_reached"
                    break
                if pages_fetched >= args.max_pages:
                    stop_reason = "max_pages_reached"
                    break
                if not raw_page_count or next_start >= total_results:
                    stop_reason = "upstream_exhausted"
                    break
                if raw_page_count < page_size:
                    stop_reason = "upstream_exhausted"
                    break
                if not new_paper_count:
                    stop_reason = "no_new_results"
                    break

                current_start = next_start
                await asyncio.sleep(3.0)

        next_offset = current_start + raw_page_count if pages_fetched else None
        has_next = (
            stop_reason in {"total_limit_reached", "max_pages_reached"}
            and next_offset is not None
            and next_offset < total_results
        )

        return {
            "ok": True,
            "source": "arxiv",
            "query": args.query,
            "arxiv_ids": args.arxiv_ids,
            "search_query": search_query,
            "returned_count": len(papers),
            "papers": papers,
            "error": None,
            "status": "partial_failed" if page_warning else "success",
            "warnings": [page_warning] if page_warning else [],
            "metadata": {
                "total_results": total_results,
                "sort": args.sort,
                "pagination": {
                    "initial_offset": args.start,
                    "page_size": args.max_results,
                    "total_limit": args.total_limit,
                    "max_pages": args.max_pages,
                    "pages_fetched": pages_fetched,
                    "raw_fetched_count": raw_fetched_count,
                    "returned_count": len(papers),
                    "has_next": has_next,
                    "next_offset": next_offset if has_next else None,
                    "stop_reason": stop_reason,
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

    except ET.ParseError as exc:
        return failure(
            code="UPSTREAM_PARSE_ERROR",
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


ARXIV_SEARCH_TOOL = AppToolSpec(
    name="arxiv_search",
    description=(
        "从 arXiv 少量检索论文并直接返回论文信息。"
        "支持主题、标题、作者、摘要、分类和 arXiv ID 精确查询。"
        "单次默认返回 5 篇，最多 20 篇。"
        "不用于批量检索、分页、文件导出或跨来源去重。"
    ),
    args_schema=ArxivSearchArgs,
    handler=arxiv_search_handler,
    requires_confirmation=False,
)
