import json
from collections.abc import Mapping
from math import ceil
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.models.system_setting import SystemSetting


SOURCE_LIMITS_KEY = "source_limits"
SOURCE_LIMIT_NAMES = ("arxiv", "dblp", "google_scholar")


def source_page_sizes() -> dict[str, int]:
    return {
        "arxiv": settings.PAPER_SEARCH_ARXIV_PAGE_SIZE,
        "dblp": settings.PAPER_SEARCH_DBLP_PAGE_SIZE,
        "google_scholar": settings.PAPER_SEARCH_GOOGLE_SCHOLAR_PAGE_SIZE,
    }


def source_limit_maxima() -> dict[str, int]:
    page_sizes = source_page_sizes()
    return {
        "arxiv": page_sizes["arxiv"] * settings.PAPER_SEARCH_ARXIV_MAX_PAGES,
        "dblp": page_sizes["dblp"] * settings.PAPER_SEARCH_DBLP_MAX_PAGES,
        "google_scholar": (
            page_sizes["google_scholar"]
            * settings.PAPER_SEARCH_GOOGLE_SCHOLAR_MAX_PAGES
        ),
    }


def default_source_limits() -> dict[str, int]:
    return {
        "arxiv": settings.PAPER_SEARCH_ARXIV_TOTAL_LIMIT,
        "dblp": settings.PAPER_SEARCH_DBLP_TOTAL_LIMIT,
        "google_scholar": settings.PAPER_SEARCH_GOOGLE_SCHOLAR_TOTAL_LIMIT,
    }


def validate_source_limits(value: Mapping[str, Any]) -> dict[str, int]:
    if set(value) != set(SOURCE_LIMIT_NAMES):
        raise ValueError("source_limits 必须包含 arxiv、dblp 和 google_scholar。")

    maxima = source_limit_maxima()
    normalized: dict[str, int] = {}
    for name in SOURCE_LIMIT_NAMES:
        limit = value[name]
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError(f"{name} 必须为整数。")
        if not 1 <= limit <= maxima[name]:
            raise ValueError(f"{name} 必须在 1 到 {maxima[name]} 之间。")
        normalized[name] = limit
    return normalized


def source_pagination_settings(
    source_limits: Mapping[str, Any],
) -> dict[str, int]:
    limits = validate_source_limits(source_limits)
    page_sizes = source_page_sizes()
    return {
        "arxiv_max_pages": ceil(limits["arxiv"] / page_sizes["arxiv"]),
        "arxiv_page_size": page_sizes["arxiv"],
        "arxiv_total_limit": limits["arxiv"],
        "dblp_max_pages": ceil(limits["dblp"] / page_sizes["dblp"]),
        "dblp_page_size": page_sizes["dblp"],
        "dblp_total_limit": limits["dblp"],
        "google_max_pages": ceil(
            limits["google_scholar"] / page_sizes["google_scholar"]
        ),
        "google_page_size": page_sizes["google_scholar"],
        "google_total_limit": limits["google_scholar"],
    }


def get_setting(db: Session, key: str, default: Any = None) -> Any:
    setting = db.query(SystemSetting).filter(SystemSetting.key == key).first()
    if setting and setting.value:
        try:
            return json.loads(setting.value)
        except json.JSONDecodeError:
            return setting.value
    return default


def save_setting(db: Session, key: str, value: Any) -> None:
    serialized = json.dumps(value, ensure_ascii=False)
    setting = db.query(SystemSetting).filter(SystemSetting.key == key).first()
    if setting is None:
        db.add(SystemSetting(key=key, value=serialized))
    else:
        setting.value = serialized
        setting.modifier_id = 0
    db.commit()


def get_source_limits(db: Session) -> dict[str, int]:
    stored = get_setting(db, SOURCE_LIMITS_KEY)
    if not isinstance(stored, Mapping):
        return default_source_limits()
    try:
        return validate_source_limits(stored)
    except ValueError:
        return default_source_limits()


def save_source_limits(
    db: Session,
    source_limits: Mapping[str, Any],
) -> dict[str, int]:
    normalized = validate_source_limits(source_limits)
    save_setting(db, SOURCE_LIMITS_KEY, normalized)
    return normalized


def get_custom_system_prompt(db: Session) -> str:
    return get_setting(db, "agent_system_prompt", "") or ""
