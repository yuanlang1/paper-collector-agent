from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.tools.search_tools.crossref.search_crossref import (
    crossref_search_handler,
)
from app.llm.tools.task_tools.search_task.args import PaperTypeCode


CrossrefSearch = Callable[
    [dict[str, Any], Any], Awaitable[dict[str, Any]]
]


def _text(value: Any) -> str | None:
    if value is None:
        return None

    value = str(value).strip()
    return value or None


def _normalize_doi(value: Any) -> str | None:
    doi = _text(value)
    if doi is None:
        return None

    return re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        doi,
        flags=re.IGNORECASE,
    ).strip() or None


def _normalize_title(value: Any) -> str:
    return re.sub(
        r"[^\w\s]",
        "",
        str(value or "").casefold(),
    ).replace(" ", "")


def _first_author(value: Any) -> str | None:
    if isinstance(value, list):
        return next(
            (
                author
                for item in value
                if (author := _text(item)) is not None
            ),
            None,
        )

    authors = _text(value)
    if authors is None:
        return None

    return re.split(r"\s*(?:,|;|\band\b)\s*", authors, 1)[0] or None


def _author_matches(expected: Any, actual: Any) -> bool:
    expected_author = _first_author(expected)
    actual_author = _first_author(actual)

    if expected_author is None or actual_author is None:
        return True

    expected_tokens = set(re.findall(r"\w+", expected_author.casefold()))
    actual_tokens = set(re.findall(r"\w+", actual_author.casefold()))
    return bool(expected_tokens & actual_tokens)


def _matching_candidate(
    paper_info: Mapping[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    title = _normalize_title(paper_info.get("title"))

    if not title:
        return None

    for candidate in candidates:
        if (
            _normalize_title(candidate.get("title")) == title
            and _author_matches(
                paper_info.get("authors"),
                candidate.get("authors"),
            )
        ):
            return candidate

    return None


def _paper_type_code(value: Any) -> int:
    try:
        return int(PaperTypeCode(str(value)))
    except (TypeError, ValueError):
        return 0


def _as_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _response_papers(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    if response.get("ok") is not True:
        error = response.get("error")
        message = (
            error.get("message")
            if isinstance(error, Mapping)
            else error
        )
        raise RuntimeError(str(message or "Crossref search failed."))

    papers = response.get("papers")
    return [paper for paper in papers if isinstance(paper, dict)] if isinstance(papers, list) else []


def _is_doi_not_found(response: Mapping[str, Any]) -> bool:
    error = response.get("error")
    return (
        isinstance(error, Mapping)
        and error.get("code") == "CROSSREF_NOT_FOUND"
    )


def _append_venue_candidate(
    paper: dict[str, Any],
    venue: str | None,
) -> None:
    if venue is None:
        return

    candidates = paper.setdefault("venue_candidates", [])
    if not isinstance(candidates, list):
        candidates = []
        paper["venue_candidates"] = candidates

    if venue not in candidates:
        candidates.append(venue)


class CrossrefMetadataEnrichmentNode:
    """Use Crossref to supplement metadata for already-selected papers."""

    def __init__(
        self,
        *,
        artifact_store: LocalArtifactStore | None = None,
        crossref_search: CrossrefSearch | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.crossref_search = crossref_search or crossref_search_handler

    async def _search(
        self,
        paper_info: Mapping[str, Any],
    ) -> tuple[dict[str, Any] | None, str]:
        doi = _normalize_doi(paper_info.get("doi"))
        if doi:
            response = await self.crossref_search({"doi": doi}, None)
            if not _is_doi_not_found(response):
                papers = _response_papers(response)
                return (
                    papers[0] if papers else None,
                    "doi",
                )

        title = _text(paper_info.get("title"))
        if title is None:
            return None, "title"

        params: dict[str, Any] = {"title": title, "limit": 5}
        first_author = _first_author(paper_info.get("authors"))
        if first_author:
            params["author"] = first_author

        response = await self.crossref_search(params, None)
        papers = _response_papers(response)
        return (
            _matching_candidate(
                paper_info,
                papers,
            ),
            "title",
        )

    def _merge(
        self,
        paper: dict[str, Any],
        crossref_paper: Mapping[str, Any],
        match_method: str,
    ) -> None:
        paper_info = paper["paper_info"]
        doi = _normalize_doi(crossref_paper.get("doi"))
        published_date = _text(crossref_paper.get("published"))
        venue_title = _text(crossref_paper.get("venue"))
        citations = max(
            _as_int(paper_info.get("citations")),
            _as_int(crossref_paper.get("citation_count")),
        )
        paper_type_code = _paper_type_code(
            crossref_paper.get("publication_type")
        )

        if not paper_info.get("doi") and doi:
            paper_info["doi"] = doi
        if not paper_info.get("publish_date") and published_date:
            paper_info["publish_date"] = published_date
        if not paper_info.get("paper_abstract"):
            paper_info["paper_abstract"] = _text(
                crossref_paper.get("abstract")
            )
        if not paper_info.get("abstract_url"):
            paper_info["abstract_url"] = _text(
                crossref_paper.get("abstract_url")
            ) or _text(crossref_paper.get("landing_url"))
        paper_info["citations"] = citations
        _append_venue_candidate(paper, venue_title)

        paper["crossref_metadata"] = {
            "status": "matched",
            "match_method": match_method,
            "doi": doi,
            "published_date": published_date,
            "citations": _as_int(crossref_paper.get("citation_count")),
            "venue_title": venue_title,
            "paper_type_code": paper_type_code,
            "crossref_type": _text(
                crossref_paper.get("publication_type")
            ),
        }

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            run_id = state.get("run_id")
            artifact_uri = state.get("enrichment_manifest_artifact_ref")
            if not isinstance(run_id, str) or not run_id:
                raise ValueError("Missing valid run_id.")
            if (
                not isinstance(artifact_uri, str)
                or not artifact_uri.startswith("artifact://")
            ):
                raise ValueError("Missing enrichment manifest artifact.")

            base_dir = self.artifact_store.base_dir.resolve()
            manifest_path = (
                base_dir / artifact_uri.removeprefix("artifact://")
            ).resolve()
            manifest_path.relative_to(base_dir)

            def read_manifest() -> dict[str, Any]:
                with manifest_path.open("r", encoding="utf-8") as file:
                    return json.load(file)

            manifest = await asyncio.to_thread(read_manifest)
            papers = manifest.get("papers", [])
            if not isinstance(papers, list):
                raise ValueError("Manifest papers must be a list.")

            matched_count = 0
            unmatched_count = 0
            warnings: list[str] = []

            for paper in papers:
                if not isinstance(paper, dict):
                    continue
                paper_info = paper.get("paper_info")
                if not isinstance(paper_info, dict):
                    continue

                title = _text(paper_info.get("title")) or ""
                try:
                    crossref_paper, match_method = await self._search(
                        paper_info
                    )
                except Exception as exc:
                    paper["crossref_metadata"] = {
                        "status": "failed",
                        "match_method": None,
                    }
                    warnings.append(f"{title} 的 Crossref 补充失败：{exc}")
                    continue

                if crossref_paper is None:
                    paper["crossref_metadata"] = {
                        "status": "not_matched",
                        "match_method": match_method,
                    }
                    unmatched_count += 1
                    continue

                self._merge(paper, crossref_paper, match_method)
                matched_count += 1

            manifest["papers"] = papers
            manifest["step_key"] = "crossref_enrichment"
            artifact = await self.artifact_store.write_json(
                run_id=run_id,
                step_key="crossref_enrichment",
                source="crossref",
                kind="paper_info_crossref_enriched_manifest_json",
                payload=manifest,
                count=len(papers),
                metadata={
                    "input_manifest": artifact_uri,
                    "matched_count": matched_count,
                    "unmatched_count": unmatched_count,
                    "failed_count": len(warnings),
                },
            )

        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"Crossref metadata enrichment failed: {exc}",
            }

        return {
            "stage": "resolving_venues",
            "status": "partial_failed" if warnings else "running",
            "degraded": bool(state.get("degraded")) or bool(warnings),
            "crossref_enrichment_manifest_artifact_ref": artifact.artifact_uri,
            "progress": {
                **state.get("progress", {}),
                "crossref_matched": matched_count,
                "crossref_unmatched": unmatched_count,
                "crossref_failed": len(warnings),
            },
            "warnings": [*state.get("warnings", []), *warnings],
            "error": None,
        }
