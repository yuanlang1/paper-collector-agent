import re


_CANONICAL_RUN_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,62}[A-Za-z0-9])?")


class InvalidRunIdError(ValueError):
    """The run ID cannot be used as a stable artifact identifier."""


def validate_run_id(value: str) -> str:
    if not _CANONICAL_RUN_ID.fullmatch(value):
        raise InvalidRunIdError(
            "run_id must be 1-64 ASCII letters, digits, '.', '_' or '-', "
            "and must start and end with a letter or digit"
        )
    return value


def validate_conversation_id(value: str) -> str:
    value = value.strip()
    if not value or len(value) > 64:
        raise ValueError("conversation_id must be 1-64 characters")
    return value


def normalize_user_id(value: str | int) -> tuple[str, int]:
    raw = str(value).strip()
    if not raw.isdecimal():
        raise ValueError("user_id must be a non-negative BIGINT")
    numeric = int(raw)
    if numeric > 9_223_372_036_854_775_807:
        raise ValueError("user_id must be a non-negative BIGINT")
    return str(numeric), numeric
