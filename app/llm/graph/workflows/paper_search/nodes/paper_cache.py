from __future__ import annotations

import asyncio
import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _normalize_text(value: Any) -> str:
    normalized = re.sub(r"\s+", " ", str(value or "").strip()).casefold()
    return re.sub(r"[^\w\s]", "", normalized).strip()


def _normalize_doi(value: Any) -> str:
    return re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        str(value or "").strip(),
        flags=re.IGNORECASE,
    ).casefold()


def identity_keys(paper_info: dict[str, Any]) -> list[str]:
    doi = _normalize_doi(paper_info.get("doi"))
    keys = [f"doi:{doi}"] if doi else []

    title = _normalize_text(paper_info.get("title"))
    authors = paper_info.get("authors")
    if isinstance(authors, list):
        first_author = _normalize_text(authors[0]) if authors else ""
    else:
        first_author = _normalize_text(
            re.split(r"\s*(?:;|\band\b)\s*", str(authors or ""), maxsplit=1,
                     flags=re.IGNORECASE)[0]
        )
    if title and first_author:
        keys.append(f"title_author:{title}|{first_author}")

    return keys


def _identity_keys(paper: dict[str, Any]) -> list[str]:
    paper_info = paper.get("paper_info") or {}
    return identity_keys(paper_info) if isinstance(paper_info, dict) else []


class PaperCacheStore:
    """Persist complete saved-paper envelopes for local metadata reuse."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()

    def _read_sync(self) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        if not self.path.exists():
            return {}, {}

        try:
            with self.path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            return {}, {}

        papers = payload.get("papers") if isinstance(payload, dict) else None
        aliases = payload.get("aliases") if isinstance(payload, dict) else None
        return (
            {
                key: value
                for key, value in (papers or {}).items()
                if isinstance(key, str) and isinstance(value, dict)
            },
            {
                key: value
                for key, value in (aliases or {}).items()
                if isinstance(key, str) and isinstance(value, str)
            },
        )

    def _write_sync(
        self,
        papers: dict[str, dict[str, Any]],
        aliases: dict[str, str],
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(
                {"version": 1, "papers": papers, "aliases": aliases},
                file,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        tmp_path.replace(self.path)

    async def upsert_saved_paper(self, paper: dict[str, Any]) -> None:
        paper_id = paper.get("paper_id")
        keys = _identity_keys(paper)
        if (
            not isinstance(paper_id, int)
            or isinstance(paper_id, bool)
            or paper_id <= 0
            or not keys
        ):
            raise ValueError("saved paper requires paper_id and identity")

        cached_paper = copy.deepcopy(paper)
        cached_paper.pop("task_paper_relation_draft", None)
        cached_paper["updated_at"] = datetime.now(timezone.utc).isoformat()

        async with self._lock:
            papers, aliases = await asyncio.to_thread(self._read_sync)
            primary_key = next(
                (aliases[key] for key in keys if key in aliases),
                keys[0],
            )
            papers[primary_key] = cached_paper
            aliases.update({key: primary_key for key in keys})
            await asyncio.to_thread(
                self._write_sync,
                papers,
                aliases,
            )

    async def get_by_paper_info(
        self,
        paper_info: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return a complete saved-paper envelope matched by DOI or title/author."""
        keys = identity_keys(paper_info)
        if not keys:
            return None

        async with self._lock:
            papers, aliases = await asyncio.to_thread(self._read_sync)
            for key in keys:
                primary_key = aliases.get(key, key)
                cached_paper = papers.get(primary_key)
                if isinstance(cached_paper, dict):
                    return copy.deepcopy(cached_paper)

        return None
