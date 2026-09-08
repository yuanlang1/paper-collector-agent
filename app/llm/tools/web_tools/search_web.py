from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import settings
from app.llm.tools.registry import Tool, ToolExecutionContext
from app.llm.tools.search_tools.common import (
    RETRYABLE_STATUS_CODES,
    clean_text,
    elapsed_ms,
)


MAX_MARKDOWN_CHARACTERS = 6_000
DOMAIN_PATTERN = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$"
)


class SearchWebArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=10)
    include_domains: list[str] = Field(default_factory=list, max_length=20)
    exclude_domains: list[str] = Field(default_factory=list, max_length=20)
    tbs: str | None = Field(default=None, min_length=1, max_length=100)
    include_markdown: bool = False

    @field_validator("query", "tbs")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("include_domains", "exclude_domains")
    @classmethod
    def validate_domains(cls, values: list[str]) -> list[str]:
        domains: list[str] = []
        for value in values:
            domain = value.strip().lower()
            if not DOMAIN_PATTERN.fullmatch(domain):
                raise ValueError("domains must be hostnames without a scheme or path")
            domains.append(domain)
        return list(dict.fromkeys(domains))

    @model_validator(mode="after")
    def validate_domain_filters(self) -> "SearchWebArgs":
        if self.include_domains and self.exclude_domains:
            raise ValueError("include_domains and exclude_domains are mutually exclusive")
        return self


class FirecrawlResponseError(ValueError):
    """Firecrawl returned a successful HTTP response with an invalid body."""


def _request_payload(args: SearchWebArgs) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": args.query,
        "limit": args.limit,
        "sources": ["web"],
    }
    if args.include_domains:
        payload["includeDomains"] = args.include_domains
    if args.exclude_domains:
        payload["excludeDomains"] = args.exclude_domains
    if args.tbs:
        payload["tbs"] = args.tbs
    if args.include_markdown:
        payload["scrapeOptions"] = {
            "formats": [{"type": "markdown"}],
            "onlyMainContent": True,
        }
    return payload


async def _post_search(
    client: httpx.AsyncClient,
    *,
    payload: dict[str, Any],
) -> dict[str, Any]:
    for attempt in range(2):
        try:
            response = await client.post(
                settings.FIRECRAWL_SEARCH_URL,
                json=payload,
            )
            if (
                response.status_code in RETRYABLE_STATUS_CODES
                and attempt == 0
            ):
                await asyncio.sleep(1)
                continue
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError as exc:
                raise FirecrawlResponseError(
                    "Firecrawl returned invalid JSON"
                ) from exc
            if not isinstance(data, Mapping):
                raise FirecrawlResponseError("Firecrawl response must be an object")
            return dict(data)
        except httpx.RequestError:
            if attempt == 1:
                raise
            await asyncio.sleep(1)

    raise RuntimeError("Firecrawl search retry loop ended unexpectedly")


def _markdown(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    return value[:MAX_MARKDOWN_CHARACTERS]


def _normalize_result(
    item: Mapping[str, Any],
    *,
    include_markdown: bool,
) -> dict[str, Any] | None:
    metadata = item.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    url = clean_text(item.get("url") or metadata.get("sourceURL"))
    if not url:
        return None

    result: dict[str, Any] = {
        "title": clean_text(item.get("title") or metadata.get("title")) or None,
        "url": url,
        "description": clean_text(
            item.get("description") or metadata.get("description")
        )
        or None,
        "status_code": metadata.get("statusCode"),
    }
    if include_markdown:
        result["markdown"] = _markdown(item.get("markdown"))
    return result


def _failure(
    *,
    args: SearchWebArgs,
    started_at: float,
    code: str,
    message: str,
    retryable: bool,
    status_code: int | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"duration_ms": elapsed_ms(started_at)}
    if status_code is not None:
        metadata["status_code"] = status_code
    return {
        "ok": False,
        "source": "firecrawl",
        "query": args.query,
        "returned_count": 0,
        "results": [],
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
        "metadata": metadata,
    }


async def search_web_handler(
    params: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    del context
    args = SearchWebArgs.model_validate(params)
    started_at = time.perf_counter()
    if not settings.FIRECRAWL_API_KEY:
        return _failure(
            args=args,
            started_at=started_at,
            code="FIRECRAWL_NOT_CONFIGURED",
            message="Firecrawl API Key 未配置。",
            retryable=False,
        )

    try:
        async with httpx.AsyncClient(
            timeout=settings.FIRECRAWL_TIMEOUT_SECONDS,
            headers={
                "Authorization": f"Bearer {settings.FIRECRAWL_API_KEY}",
                "Content-Type": "application/json",
            },
        ) as client:
            payload = await _post_search(
                client,
                payload=_request_payload(args),
            )

        if payload.get("success") is not True:
            raise FirecrawlResponseError("Firecrawl search was not successful")
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise FirecrawlResponseError("Firecrawl response is missing data")
        raw_results = data.get("web")
        if not isinstance(raw_results, list):
            raise FirecrawlResponseError("Firecrawl response is missing web results")
        results = [
            result
            for item in raw_results
            if isinstance(item, Mapping)
            if (
                result := _normalize_result(
                    item,
                    include_markdown=args.include_markdown,
                )
            ) is not None
        ]
        return {
            "ok": True,
            "source": "firecrawl",
            "query": args.query,
            "returned_count": len(results),
            "results": results,
            "warning": clean_text(payload.get("warning")) or None,
            "error": None,
            "metadata": {
                "search_id": clean_text(payload.get("id")) or None,
                "credits_used": payload.get("creditsUsed"),
                "include_markdown": args.include_markdown,
                "duration_ms": elapsed_ms(started_at),
            },
        }
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        if status_code in {401, 403}:
            code, message, retryable = (
                "FIRECRAWL_AUTH_ERROR",
                "Firecrawl API Key 无效或无权访问。",
                False,
            )
        elif status_code == 402:
            code, message, retryable = (
                "FIRECRAWL_CREDITS_EXHAUSTED",
                "Firecrawl 额度不足。",
                False,
            )
        elif status_code == 429:
            code, message, retryable = (
                "FIRECRAWL_RATE_LIMITED",
                "Firecrawl 请求过于频繁。",
                True,
            )
        else:
            code, message, retryable = (
                "FIRECRAWL_HTTP_ERROR",
                f"Firecrawl 返回 HTTP {status_code}。",
                status_code in RETRYABLE_STATUS_CODES,
            )
        return _failure(
            args=args,
            started_at=started_at,
            code=code,
            message=message,
            retryable=retryable,
            status_code=status_code,
        )
    except httpx.RequestError:
        return _failure(
            args=args,
            started_at=started_at,
            code="FIRECRAWL_REQUEST_ERROR",
            message="无法连接 Firecrawl。",
            retryable=True,
        )
    except FirecrawlResponseError as exc:
        return _failure(
            args=args,
            started_at=started_at,
            code="FIRECRAWL_RESPONSE_INVALID",
            message=str(exc),
            retryable=False,
        )


SEARCH_WEB_TOOL = Tool(
    name="search_web",
    description=(
        "通过 Firecrawl 搜索实时网页、官网文档或新闻。"
        "默认只返回标题、链接和摘要；仅在需要阅读页面内容时设置 include_markdown=true。"
        "完整论文检索仍应使用 paper_search_agent。"
    ),
    input_schema=SearchWebArgs.model_json_schema(),
    fn=search_web_handler,
    requires_confirmation=False,
)
