from __future__ import annotations

from pathlib import Path


SOUL_PATH = Path(__file__).with_name("soul.md")


class SoulLoadError(RuntimeError):
    """The main Agent soul cannot be loaded safely."""


def load_soul() -> str:
    try:
        content = SOUL_PATH.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SoulLoadError(f"Unable to load Agent soul: {SOUL_PATH}") from exc

    if not content:
        raise SoulLoadError(f"Agent soul is empty: {SOUL_PATH}")

    return content
