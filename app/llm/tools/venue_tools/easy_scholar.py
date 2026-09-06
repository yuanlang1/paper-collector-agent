from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any, Dict

import httpx
from sqlalchemy.orm import Session

from app.config import settings
from app.llm.tools.registry import Tool
from app.llm.tools.search_tools.common import (
    RETRYABLE_STATUS_CODES,
    elapsed_ms,
)
from app.llm.tools.venue_tools.easy_scholar_args import (
    EasyScholarVenueArgs,
)

EASY_SCHOLAR_API_URL = settings.EASY_SCHOLAR_URL

_rate_limit_lock = asyncio.Lock()
_last_request_at = 0.0

RANK_FIELD_BY_INDEX = {
    1: "oneRankText",
    2: "twoRankText",
    3: "threeRankText",
    4: "fourRankText",
    5: "fiveRankText",
}
RANK_TEXT_FIELDS = tuple(RANK_FIELD_BY_INDEX.values())


def _text(value: Any) -> str | None:
    if value is None:
        return None

    result = str(value).strip()
    return result or None


def _merge_official_rank(
    data: Mapping[str, Any],
) -> dict[str, Any]:
    official_rank = data.get("officialRank")

    if not isinstance(official_rank, Mapping):
        return {}

    all_ranks = official_rank.get("all")
    selected_ranks = official_rank.get("select")

    return {
        **(dict(all_ranks) if isinstance(all_ranks, Mapping) else {}),
        **(
            dict(selected_ranks)
            if isinstance(selected_ranks, Mapping)
            else {}
        ),
    }


def _parse_custom_rank(
    data: Mapping[str, Any],
) -> list[dict[str, Any]]:
    custom_rank = data.get("customRank")

    if not isinstance(custom_rank, Mapping):
        return []

    rank_info_list = custom_rank.get("rankInfo")
    rank_values = custom_rank.get("rank")

    if not isinstance(rank_info_list, list) or not isinstance(rank_values, list):
        return []

    rank_info_by_uuid = {
        str(item["uuid"]): item
        for item in rank_info_list
        if isinstance(item, Mapping) and item.get("uuid") is not None
    }
    resolved_ranks: list[dict[str, Any]] = []

    for value in rank_values:
        if not isinstance(value, str):
            continue

        uuid, separator, rank_index_text = value.partition("&&&")

        if not separator:
            continue

        try:
            rank_index = int(rank_index_text)
        except ValueError:
            continue

        rank_field = RANK_FIELD_BY_INDEX.get(rank_index)
        rank_info = rank_info_by_uuid.get(uuid)

        if rank_info is None or rank_field is None:
            continue

        rank_text = {
            field: text
            for field in RANK_TEXT_FIELDS
            if (text := _text(rank_info.get(field))) is not None
        }
        current_rank = rank_text.get(rank_field)

        if current_rank is None:
            continue

        resolved_ranks.append(
            {
                "uuid": uuid,
                "abbName": _text(rank_info.get("abbName")),
                "rankText": rank_text,
                "currentRank": current_rank,
            }
        )

    return resolved_ranks


async def _wait_for_rate_limit() -> None:
    global _last_request_at

    async with _rate_limit_lock:
        elapsed = time.monotonic() - _last_request_at
        wait_seconds = max(0.0, 0.5 - elapsed)

        if wait_seconds:
            await asyncio.sleep(wait_seconds)

        _last_request_at = time.monotonic()


def _to_result(
    *,
    publication_name: str,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "publication_name": (
            _text(data.get("publicationName"))
            or _text(data.get("name"))
            or publication_name
        ),
        "abbreviation": (
            _text(data.get("abbreviation"))
            or _text(data.get("abbr"))
        ),
        "official_rank": _merge_official_rank(data),
        "custom_rank": _parse_custom_rank(data),
    }


async def easy_scholar_venue_handler(
    params: Dict[str, Any],
    _db: Session,
) -> Dict[str, Any]:
    del _db

    args = EasyScholarVenueArgs.model_validate(params)
    started_at = time.perf_counter()

    if not settings.EASY_SCHOLAR_SECRET_KEY:
        return {
            "ok": False,
            "source": "easy_scholar",
            "publication_name": args.publication_name,
            "data": None,
            "error": {
                "code": "CONFIGURATION_ERROR",
                "message": "EasyScholar 密钥未配置。",
                "retryable": False,
            },
            "metadata": {
                "duration_ms": elapsed_ms(started_at),
            },
        }

    try:
        await _wait_for_rate_limit()

        async with httpx.AsyncClient(
            timeout=settings.EASY_SCHOLAR_TIMEOUT_SECONDS,
        ) as client:
            response = await client.get(
                EASY_SCHOLAR_API_URL,
                params={
                    "secretKey": settings.EASY_SCHOLAR_SECRET_KEY,
                    "publicationName": args.publication_name,
                },
            )

        response.raise_for_status()
        payload = response.json()

        if payload.get("code") != 200:
            return {
                "ok": False,
                "source": "easy_scholar",
                "publication_name": args.publication_name,
                "data": None,
                "error": {
                    "code": "UPSTREAM_BUSINESS_ERROR",
                    "message": str(payload.get("msg") or "查询失败"),
                    "retryable": False,
                },
                "metadata": {
                    "duration_ms": elapsed_ms(started_at),
                    "upstream_code": payload.get("code"),
                },
            }

        data = payload.get("data")
        if not isinstance(data, Mapping):
            return {
                "ok": False,
                "source": "easy_scholar",
                "publication_name": args.publication_name,
                "data": None,
                "error": {
                    "code": "UPSTREAM_EMPTY_RESULT",
                    "message": "未找到该期刊的等级信息。",
                    "retryable": False,
                },
                "metadata": {
                    "duration_ms": elapsed_ms(started_at),
                },
            }

        return {
            "ok": True,
            "source": "easy_scholar",
            "publication_name": args.publication_name,
            "data": _to_result(
                publication_name=args.publication_name,
                data=data,
            ),
            "error": None,
            "metadata": {
                "duration_ms": elapsed_ms(started_at),
            },
        }

    except httpx.HTTPStatusError as exc:
        return {
            "ok": False,
            "source": "easy_scholar",
            "publication_name": args.publication_name,
            "data": None,
            "error": {
                "code": "UPSTREAM_HTTP_ERROR",
                "message": f"EasyScholar 返回 HTTP {exc.response.status_code}",
                "retryable": (
                    exc.response.status_code in RETRYABLE_STATUS_CODES
                ),
            },
            "metadata": {
                "duration_ms": elapsed_ms(started_at),
                "status_code": exc.response.status_code,
            },
        }

    except httpx.RequestError:
        return {
            "ok": False,
            "source": "easy_scholar",
            "publication_name": args.publication_name,
            "data": None,
            "error": {
                "code": "UPSTREAM_REQUEST_ERROR",
                "message": "无法连接 EasyScholar。",
                "retryable": True,
            },
            "metadata": {
                "duration_ms": elapsed_ms(started_at),
            },
        }

    except ValueError as exc:
        return {
            "ok": False,
            "source": "easy_scholar",
            "publication_name": args.publication_name,
            "data": None,
            "error": {
                "code": "UPSTREAM_PARSE_ERROR",
                "message": str(exc),
                "retryable": False,
            },
            "metadata": {
                "duration_ms": elapsed_ms(started_at),
            },
        }


EASY_SCHOLAR_VENUE_TOOL = Tool(
    name="get_venue_info",
    description=(
        "查询单个学术期刊的等级信息。可返回 JCR 分区、CCF、"
        "影响因子、中科院分区、北大核心、CSSCI、EI、CSCD 等；"
        "仅用于期刊等级查询，不用于检索论文。"
    ),
    input_schema=EasyScholarVenueArgs.model_json_schema(),
    fn=easy_scholar_venue_handler,
    requires_confirmation=False,
)
