import re
import time
from typing import Any


RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []

    return value if isinstance(value, list) else [value]


def parse_year(value: Any) -> int | None:
    if value is None:
        return None

    match = re.search(
        r"\b(19|20)\d{2}\b",
        str(value),
    )
    return int(match.group(0)) if match else None


def elapsed_ms(started_at: float) -> int:
    return round((time.perf_counter() - started_at) * 1000)


def is_retryable_status(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS_CODES