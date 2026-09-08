from __future__ import annotations

import asyncio
import html
import re
import time
from collections.abc import Mapping
from typing import Any, Dict
from urllib.parse import quote

import httpx
from sqlalchemy.orm import Session

from app.config import settings
from app.llm.tools.registry import Tool
from app.llm.tools.search_tools.common import (
    RETRYABLE_STATUS_CODES,
    clean_text,
    elapsed_ms,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

from datetime import date

CROSSREF_API_URL = settings.CROSSREF_API_URL.rstrip("/")

class CrossrefSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doi: str | None = Field(None, min_length=1, max_length=500)
    query: str | None = Field(None, min_length=1, max_length=500)
    title: str | None = Field(None, min_length=1, max_length=500)
    author: str | None = Field(None, min_length=1, max_length=200)
    issn: str | None = Field(None, min_length=4, max_length=20)
    from_pub_date: date | None = None
    until_pub_date: date | None = None
    work_type: str | None = Field(None, min_length=1, max_length=80)
    limit: int = Field(5, ge=1, le=20)

    @model_validator(mode="after")
    def validate_request(self) -> "CrossrefSearchArgs":
        search_fields = [
            self.query,
            self.title,
            self.author,
            self.issn,
            self.from_pub_date,
            self.until_pub_date,
            self.work_type,
        ]

        if self.doi and any(value is not None for value in search_fields):
            raise ValueError(
                "doi cannot be combined with Crossref search filters"
            )

        if not self.doi and not any(
            value is not None for value in search_fields
        ):
            raise ValueError(
                "doi, query, title, author, issn, date, or work_type is required"
            )

        if (
            self.from_pub_date is not None
            and self.until_pub_date is not None
            and self.from_pub_date > self.until_pub_date
        ):
            raise ValueError(
                "from_pub_date must be less than or equal to until_pub_date"
            )

        return self


class CrossrefResponseParseError(Exception):
    """Crossref returned an invalid metadata response."""


def _normalize_doi(value: str) -> str:
    return re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        value.strip(),
        flags=re.IGNORECASE,
    ).strip()


def _clean_abstract(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    text = html.unescape(re.sub(r"<[^>]+>", " ", value))
    return clean_text(text) or None


def _authors(item: Mapping[str, Any]) -> list[str]:
    authors: list[str] = []

    for author in item.get("author") or []:
        if not isinstance(author, Mapping):
            continue

        name = clean_text(
            " ".join(
                value
                for value in [
                    str(author.get("given") or "").strip(),
                    str(author.get("family") or "").strip(),
                ]
                if value
            )
        )
        if name:
            authors.append(name)

    return authors


def _published_date(item: Mapping[str, Any]) -> str | None:
    for field in ("published-print", "published-online", "issued"):
        date_part = item.get(field)
        if not isinstance(date_part, Mapping):
            continue

        date_parts = date_part.get("date-parts")
        if not isinstance(date_parts, list) or not date_parts:
            continue

        values = date_parts[0]
        if not isinstance(values, list) or not values:
            continue

        try:
            year = int(values[0])
            month = int(values[1]) if len(values) > 1 else 1
            day = int(values[2]) if len(values) > 2 else 1
            return f"{year:04d}-{month:02d}-{day:02d}"
        except (TypeError, ValueError):
            continue

    return None


def _pdf_url(item: Mapping[str, Any]) -> str | None:
    links = item.get("link")

    if not isinstance(links, list):
        return None

    for link in links:
        if not isinstance(link, Mapping):
            continue

        if str(link.get("content-type") or "").lower() != "application/pdf":
            continue

        url = clean_text(link.get("URL"))
        if url:
            return url

    return None


def _normalize_work(item: Mapping[str, Any]) -> dict[str, Any] | None:
    titles = item.get("title")
    title = (
        clean_text(titles[0])
        if isinstance(titles, list) and titles
        else ""
    )
    doi = clean_text(item.get("DOI"))

    if not title or not doi:
        return None

    venue_titles = item.get("container-title")
    venue = (
        clean_text(venue_titles[0])
        if isinstance(venue_titles, list) and venue_titles
        else None
    )
    landing_url = clean_text(item.get("URL")) or f"https://doi.org/{doi}"

    try:
        citation_count = max(
            int(item.get("is-referenced-by-count") or 0),
            0,
        )
    except (TypeError, ValueError):
        citation_count = 0

    return {
        "source": "crossref",
        "source_id": doi,
        "title": title,
        "authors": _authors(item),
        "abstract": _clean_abstract(item.get("abstract")),
        "published": _published_date(item),
        "year": None,
        "updated": None,
        "venue": venue,
        "publication_type": clean_text(item.get("type")) or None,
        "doi": doi,
        "citation_count": citation_count,
        "landing_url": landing_url,
        "abstract_url": landing_url,
        "pdf_url": _pdf_url(item),
        "publisher": clean_text(item.get("publisher")) or None,
        "issn": item.get("ISSN") if isinstance(item.get("ISSN"), list) else [],
    }


def _request_params(args: CrossrefSearchArgs) -> dict[str, str | int]:
    params: dict[str, str | int] = {"rows": args.limit}

    if args.title:
        params["query.bibliographic"] = args.title
    elif args.query:
        params["query"] = args.query

    if args.author:
        params["query.author"] = args.author

    filters: list[str] = []
    if args.issn:
        filters.append(f"issn:{args.issn}")
    if args.from_pub_date:
        filters.append(f"from-pub-date:{args.from_pub_date.isoformat()}")
    if args.until_pub_date:
        filters.append(f"until-pub-date:{args.until_pub_date.isoformat()}")
    if args.work_type:
        filters.append(f"type:{args.work_type}")
    if filters:
        params["filter"] = ",".join(filters)

    if settings.CROSSREF_MAILTO:
        params["mailto"] = settings.CROSSREF_MAILTO

    return params


async def _fetch_crossref(
    *,
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, str | int],
) -> dict[str, Any]:
    for attempt in range(3):
        try:
            response = await client.get(url, params=params)

            if (
                response.status_code in RETRYABLE_STATUS_CODES
                and attempt < 2
            ):
                await asyncio.sleep(2**attempt)
                continue

            response.raise_for_status()

            try:
                payload = response.json()
            except ValueError as exc:
                raise CrossrefResponseParseError(
                    "Crossref returned invalid JSON"
                ) from exc

            if not isinstance(payload, Mapping):
                raise CrossrefResponseParseError(
                    "Crossref response must be an object"
                )

            return dict(payload)

        except httpx.RequestError:
            if attempt == 2:
                raise
            await asyncio.sleep(2**attempt)

    raise RuntimeError("Crossref request failed")


def _query_label(args: CrossrefSearchArgs) -> str:
    return (
        args.doi
        or args.title
        or args.query
        or args.author
        or args.issn
        or "Crossref search"
    )


async def crossref_search_handler(
    params: Dict[str, Any],
    _db: Session,
) -> Dict[str, Any]:
    del _db

    args = CrossrefSearchArgs.model_validate(params)
    started_at = time.perf_counter()
    query = _query_label(args)

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
            "source": "crossref",
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
        headers = {
            "User-Agent": (
                "paper-collector-agent/1.0 "
                f"(mailto:{settings.CROSSREF_MAILTO or 'not-configured'})"
            ),
        }

        async with httpx.AsyncClient(
            timeout=settings.CROSSREF_TIMEOUT_SECONDS,
            headers=headers,
        ) as client:
            if args.doi:
                doi = _normalize_doi(args.doi)
                payload = await _fetch_crossref(
                    client=client,
                    url=f"{CROSSREF_API_URL}/works/{quote(doi, safe='')}",
                    params=(
                        {"mailto": settings.CROSSREF_MAILTO}
                        if settings.CROSSREF_MAILTO
                        else {}
                    ),
                )
                message = payload.get("message")
                items = [message] if isinstance(message, Mapping) else []
                total_results = len(items)
            else:
                payload = await _fetch_crossref(
                    client=client,
                    url=f"{CROSSREF_API_URL}/works",
                    params=_request_params(args),
                )
                message = payload.get("message")
                if not isinstance(message, Mapping):
                    raise CrossrefResponseParseError(
                        "Crossref response is missing message"
                    )
                raw_items = message.get("items")
                items = raw_items if isinstance(raw_items, list) else []
                total_results = int(message.get("total-results") or 0)

        papers = [
            paper
            for item in items
            if isinstance(item, Mapping)
            if (paper := _normalize_work(item)) is not None
        ]

        return {
            "ok": True,
            "source": "crossref",
            "query": query,
            "returned_count": len(papers),
            "papers": papers,
            "error": None,
            "metadata": {
                "total_results": total_results,
                "duration_ms": elapsed_ms(started_at),
            },
        }

    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404 and args.doi:
            return failure(
                code="CROSSREF_NOT_FOUND",
                message="Crossref 未找到该 DOI。",
                retryable=False,
                status_code=404,
            )

        return failure(
            code="UPSTREAM_HTTP_ERROR",
            message=f"Crossref returned HTTP {exc.response.status_code}",
            retryable=exc.response.status_code in RETRYABLE_STATUS_CODES,
            status_code=exc.response.status_code,
        )

    except httpx.RequestError:
        return failure(
            code="UPSTREAM_REQUEST_ERROR",
            message="无法连接 Crossref。",
            retryable=True,
        )

    except CrossrefResponseParseError as exc:
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


CROSSREF_SEARCH_TOOL = Tool(
    name="crossref_search",
    description=(
        "通过 Crossref 查询论文元数据。支持 DOI 精确查询，或按标题、"
        "关键词、作者、ISSN、发表日期和作品类型检索。"
        "返回 DOI、作者、期刊或会议、发表日期、引用数及可用 PDF 链接。"
    ),
    input_schema=CrossrefSearchArgs.model_json_schema(),
    fn=crossref_search_handler,
    requires_confirmation=False,
)
