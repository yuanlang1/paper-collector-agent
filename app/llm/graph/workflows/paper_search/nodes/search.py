from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy.orm import Session

from app.llm.tools.search_tools.google_scholar.search_google_scholar import (
    google_scholar_search_service,
)


SearchHandler = Callable[
    [dict[str, Any], Session | None],
    Awaitable[dict[str, Any]],
]

SEARCH_HANDLERS: dict[str, SearchHandler] = {
    "Google Scholar": google_scholar_search_service,
}


def _source_step_key(source: str) -> str:
    return "search_" + source.lower().replace(" ", "_").replace("-", "_")


def _error_message(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        message = value.get("message") or value.get("code")
        return str(message) if message else str(value)
    return str(value)
