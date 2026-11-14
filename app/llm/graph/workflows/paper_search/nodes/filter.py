from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping
from datetime import date
from typing import Any

from app.infrastructure.paper_service.paper_client import paper_service_client
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.paper_search.paper_search_schemas import (
    PromptUnderstandingArgs,
    SearchTagArgs,
)


def _text(value: Any) -> str | None:
    if value is None:
        return None

    normalized = str(value).strip()
    return normalized or None


def _normalize_title(value: str) -> str:
    normalized = value.casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return normalized.strip()


def _normalize_doi(value: Any) -> str | None:
    doi = _text(value)

    if not doi:
        return None

    doi = re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        doi,
        flags=re.IGNORECASE,
    )

    return doi.casefold()


def _normalize_source(value: Any) -> str:
    source = (_text(value) or "").casefold()

    mapping = {
        "arxiv": "arXiv",
        "dblp": "DBLP",
        "crossref": "Crossref",
        "google_scholar": "Google Scholar",
        "google scholar": "Google Scholar",
    }

    return mapping.get(source, _text(value) or "Unknown")


def _parse_publish_date(value: Any) -> str | None:
    value = _text(value)

    if not value:
        return None

    match = re.match(r"^\d{4}-\d{2}-\d{2}", value)

    if not match:
        return None

    try:
        return date.fromisoformat(match.group(0)).isoformat()
    except ValueError:
        return None


def _published_at(paper_info: Mapping[str, Any]) -> date:
    published_date = _parse_publish_date(paper_info.get("publish_date"))
    return date.fromisoformat(published_date) if published_date else date.min


def _parse_year(value: Any) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", str(value or ""))
    return int(match.group(0)) if match else None


def _as_authors(value: Any) -> str | None:
    if isinstance(value, list):
        authors = [str(author).strip() for author in value if str(author).strip()]
        return ", ".join(authors) or None

    return _text(value)


def _author_names(value: Any) -> list[str]:
    if isinstance(value, list):
        return [name for author in value if (name := _text(author))]
    return [
        name.strip()
        for name in re.split(
            r"[,，;；]|\band\b", _text(value) or "", flags=re.IGNORECASE
        )
        if name.strip()
    ]


def _as_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _normalize_author(value: str) -> str:
    normalized = value.casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return normalized.strip()


def _first_author(value: Any) -> str | None:
    if isinstance(value, list):
        return next(
            (_text(author) for author in value if _text(author)),
            None,
        )

    author_text = _text(value)
    if not author_text:
        return None

    return (
        re.split(
            r"\s*(?:;|\band\b)\s*",
            author_text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        or None
    )


def _identity_keys(paper: dict[str, Any]) -> tuple[str | None, str | None]:
    doi = _normalize_doi(paper.get("doi"))
    title = _text(paper.get("title"))
    first_author = _first_author(paper.get("authors"))

    title_author_key = None
    if title and first_author:
        title_author_key = (
            f"title_author:{_normalize_title(title)}|{_normalize_author(first_author)}"
        )

    doi_key = f"doi:{doi}" if doi else None
    return doi_key, title_author_key


def _matches_filters(
    paper: dict[str, Any],
    understanding: PromptUnderstandingArgs,
) -> bool:
    paper_year = _parse_year(paper.get("published")) or _parse_year(paper.get("year"))

    if (
        understanding.yearFrom is not None
        and paper_year is not None
        and paper_year < understanding.yearFrom
    ):
        return False

    if (
        understanding.yearTo is not None
        and paper_year is not None
        and paper_year > understanding.yearTo
    ):
        return False

    return True


def _to_paper_info_draft(
    paper: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    title = _text(paper.get("title"))

    if not title:
        raise ValueError("论文缺少 title。")

    if len(title) > 200:
        raise ValueError("论文 title 超过 paper_info.title 最大长度 200。")

    authors = _as_authors(paper.get("authors"))

    if authors and len(authors) > 1000:
        raise ValueError("论文 authors 超过 paper_info.authors 最大长度 1000。")

    abstract_url = _text(paper.get("abstract_url")) or _text(paper.get("landing_url"))

    if abstract_url and len(abstract_url) > 200:
        abstract_url = None

    pdf_url = _text(paper.get("pdf_url"))

    if pdf_url and len(pdf_url) > 500:
        raise ValueError("论文 pdf_url 超过 paper_info.pdf_url 最大长度 500。")

    published_date = _parse_publish_date(paper.get("published"))
    year = _parse_year(paper.get("year")) or _parse_year(paper.get("published"))

    return {
        "title": title,
        "authors": authors,
        "publish_date": published_date or (f"{year}-01-01" if year else None),
        "publish_date_precision": "day" if published_date else "year" if year else None,
        "paper_abstract": _text(paper.get("abstract")),
        "abstract_source": paper.get("abstract_source"),
        "ai_abstract": None,
        "doi": _normalize_doi(paper.get("doi")),
        "venue_id": None,
        "citations": _as_int(paper.get("citation_count")),
        "keywords": None,
        "source": source,
        "pdf_url": pdf_url,
        "abstract_url": abstract_url,
    }


def _merge_envelope(
    current: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    current_info = current["paper_info"]
    incoming_info = incoming["paper_info"]

    incoming_abstract = incoming_info.get("paper_abstract")
    current_abstract = current_info.get("paper_abstract")
    current_is_snippet = current_info.get("abstract_source") == "search_snippet"
    incoming_is_snippet = incoming_info.get("abstract_source") == "search_snippet"
    if incoming_abstract and (
        not current_abstract
        or (current_is_snippet and not incoming_is_snippet)
        or (
            current_is_snippet == incoming_is_snippet
            and len(incoming_abstract) > len(current_abstract)
        )
    ):
        current_info["paper_abstract"] = incoming_info["paper_abstract"]
        current_info["abstract_source"] = incoming_info.get("abstract_source")

    if not current_info.get("doi") and incoming_info["doi"]:
        current_info["doi"] = incoming_info["doi"]

    if not current_info.get("authors") and incoming_info["authors"]:
        current_info["authors"] = incoming_info["authors"]

    if incoming_info["publish_date"] and (
        not current_info.get("publish_date")
        or (
            current_info.get("publish_date_precision") == "year"
            and incoming_info.get("publish_date_precision") == "day"
        )
    ):
        current_info["publish_date"] = incoming_info["publish_date"]
        current_info["publish_date_precision"] = incoming_info.get(
            "publish_date_precision"
        )

    if not current_info.get("abstract_url") and incoming_info["abstract_url"]:
        current_info["abstract_url"] = incoming_info["abstract_url"]

    if not current_info.get("pdf_url") and incoming_info["pdf_url"]:
        current_info["pdf_url"] = incoming_info["pdf_url"]

    current_info["citations"] = max(
        current_info.get("citations") or 0,
        incoming_info.get("citations") or 0,
    )

    source_refs: list[dict[str, Any]] = []
    seen_source_refs: set[str] = set()

    for source_ref in [
        *current.get("source_refs", []),
        *incoming.get("source_refs", []),
    ]:
        source_ref_key = json.dumps(
            source_ref,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

        if source_ref_key not in seen_source_refs:
            source_refs.append(source_ref)
            seen_source_refs.add(source_ref_key)

    current["source_refs"] = source_refs
    current["pdf_candidates"] = list(
        dict.fromkeys(
            [
                *current.get("pdf_candidates", []),
                *incoming.get("pdf_candidates", []),
            ]
        )
    )
    current["venue_candidates"] = list(
        dict.fromkeys(
            [
                *current.get("venue_candidates", []),
                *incoming.get("venue_candidates", []),
            ]
        )
    )

    return current


class NormalizeDeduplicateFilterNode:
    def __init__(
        self,
        artifact_store: LocalArtifactStore | None = None,
        paper_lookup: Callable[
            [list[dict[str, Any]]],
            Awaitable[dict[str, Any]],
        ]
        | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.paper_lookup = paper_lookup or paper_service_client.batch_lookup_papers

    @staticmethod
    def _normalize_payloads(
        payloads: list[dict[str, Any]],
        understanding: PromptUnderstandingArgs,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]], int]:
        papers_by_doi: dict[str, dict[str, Any]] = {}
        papers_by_title_author: dict[str, dict[str, Any]] = {}
        normalized_papers: list[dict[str, Any]] = []
        identity_conflicts: list[dict[str, str]] = []
        rejected: list[dict[str, str]] = []
        filtered_count = 0

        for payload in payloads:
            source = _normalize_source(payload.get("source"))

            for paper in payload.get("papers", []):
                if not isinstance(paper, dict):
                    continue

                if not _matches_filters(
                    paper,
                    understanding,
                ):
                    filtered_count += 1
                    continue

                try:
                    paper_info = _to_paper_info_draft(
                        paper=paper,
                        source=source,
                    )

                    doi_key, title_author_key = _identity_keys(paper)
                    author_names = _author_names(paper.get("authors"))
                    if not doi_key and (not title_author_key or not author_names):
                        raise ValueError(
                            "论文缺少 DOI，或缺少标题与第一作者，无法安全去重。"
                        )
                except ValueError as exc:
                    rejected.append(
                        {
                            "title": str(paper.get("title") or ""),
                            "reason": str(exc),
                        }
                    )
                    continue

                envelope = {
                    "dedup_key": doi_key or title_author_key,
                    "author_names": author_names,
                    "paper_info": paper_info,
                    "source_refs": [
                        {
                            "source": source,
                            "source_id": paper.get("source_id"),
                            "doi": paper.get("doi"),
                            "arxiv_id": paper.get("arxiv_id"),
                            "landing_url": (
                                paper.get("landing_url") or paper.get("abstract_url")
                            ),
                        }
                    ],
                    "venue_candidates": [
                        value
                        for value in [
                            _text(paper.get("venue")),
                            _text(paper.get("journal_ref")),
                        ]
                        if value
                    ],
                    "pdf_candidates": [
                        value
                        for value in [
                            _text(paper.get("pdf_url")),
                        ]
                        if value
                    ],
                }

                doi_match = papers_by_doi.get(doi_key) if doi_key else None
                title_author_match = (
                    papers_by_title_author.get(title_author_key)
                    if title_author_key
                    else None
                )

                incoming_doi = paper_info["doi"]
                matched_doi = (
                    title_author_match["paper_info"].get("doi")
                    if title_author_match
                    else None
                )

                if (
                    title_author_match
                    and incoming_doi
                    and matched_doi
                    and incoming_doi != matched_doi
                ):
                    keep_incoming_doi = _published_at(paper_info) > _published_at(
                        title_author_match["paper_info"]
                    )
                    if keep_incoming_doi:
                        old_doi_key = f"doi:{matched_doi}"
                        if papers_by_doi.get(old_doi_key) is title_author_match:
                            papers_by_doi.pop(old_doi_key)
                        title_author_match["paper_info"]["doi"] = incoming_doi

                    identity_conflicts.append(
                        {
                            "title": paper_info["title"],
                            "title_author_key": title_author_key or "",
                            "existing_doi": matched_doi,
                            "incoming_doi": incoming_doi,
                            "selected_doi": (
                                incoming_doi if keep_incoming_doi else matched_doi
                            ),
                        }
                    )

                target = doi_match or title_author_match

                if target:
                    target = _merge_envelope(target, envelope)
                    if not target["author_names"]:
                        target["author_names"] = envelope["author_names"]
                else:
                    target = envelope
                    normalized_papers.append(target)

                resolved_doi = _normalize_doi(target["paper_info"].get("doi"))
                if resolved_doi:
                    resolved_doi_key = f"doi:{resolved_doi}"
                    target["dedup_key"] = resolved_doi_key
                    papers_by_doi[resolved_doi_key] = target

                if title_author_key:
                    existing = papers_by_title_author.get(title_author_key)
                    if existing is None or existing is target:
                        papers_by_title_author[title_author_key] = target
        return normalized_papers, identity_conflicts, rejected, filtered_count

    async def _lookup_existing_papers(
        self,
        papers: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        queries = []
        for paper in papers:
            info = paper["paper_info"]
            if info["doi"]:
                queries.append({"doi": info["doi"]})
            else:
                queries.append(
                    {"title": info["title"], "authors": paper["author_names"]}
                )

        if not queries:
            return [], []

        lookup = await self.paper_lookup(queries)
        if lookup["ok"] is not True:
            raise ValueError(lookup["error"] or "批量论文查询失败。")

        new_papers = []
        existing_by_id: dict[int, dict[str, Any]] = {}
        for candidate, result in zip(papers, lookup["result"]["results"], strict=True):
            info = candidate["paper_info"]
            if not result["status"]:
                new_papers.append(candidate)
                continue

            matched = result["paper"]
            paper_id = matched["paper_id"]
            existing = existing_by_id.get(paper_id)
            if existing is not None:
                refs = existing["source_refs"]
                refs.extend(ref for ref in candidate["source_refs"] if ref not in refs)
                continue

            venue = matched.get("venue_info")
            authors = [author for author in matched["authors"] if author]
            abstract = matched.get("abstract")
            candidate.update(
                paper_id=paper_id,
                paper_info={
                    **info,
                    "title": matched["title"],
                    "authors": _as_authors(authors) or info["authors"],
                    "doi": matched.get("doi") or info["doi"],
                    "paper_abstract": abstract or info["paper_abstract"],
                    "abstract_source": "database"
                    if abstract
                    else info.get("abstract_source"),
                    "ai_abstract": matched.get("ai_abstract"),
                    "pdf_url": matched.get("pdf_url") or info["pdf_url"],
                    "venue_id": venue["id"] if venue else None,
                },
                venue_resolution={
                    "status": "resolved",
                    "source": "database",
                    "venue_id": venue["id"],
                    "venue_draft": venue,
                }
                if venue
                else None,
            )
            existing_by_id[paper_id] = candidate

        return new_papers, list(existing_by_id.values())

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            child_run_id = state["child_run_id"]

            understanding = PromptUnderstandingArgs.model_validate(
                state.get("query_understanding")
            )
            search_tag = SearchTagArgs.model_validate(state.get("search_tag"))
            artifact_uris = state.get(
                "raw_result_artifact_refs",
                [],
            )

            if not isinstance(artifact_uris, list) or not artifact_uris:
                raise ValueError("缺少分源检索结果 artifact。")

            payloads = []
            for uri in artifact_uris:
                payloads.append(await self.artifact_store.read_json_uri(uri))

            normalized_papers, identity_conflicts, rejected, filtered_count = (
                self._normalize_payloads(payloads, understanding)
            )

            new_papers, existing_papers = await self._lookup_existing_papers(
                normalized_papers
            )

            existing_artifact = await self.artifact_store.write_json(
                run_id=child_run_id,
                step_key="find_existing_papers",
                source="database",
                payload={
                    "run_id": child_run_id,
                    "step_key": "find_existing_papers",
                    "papers": existing_papers,
                },
            )

            source_stats = {
                source: dict(summary)
                for source, summary in (state.get("provider_stats") or {}).items()
                if isinstance(source, str) and isinstance(summary, dict)
            }
            for source, summary in source_stats.items():
                summary["accepted_count"] = sum(
                    1
                    for paper in [*new_papers, *existing_papers]
                    if any(
                        ref.get("source") == _normalize_source(source)
                        for ref in paper.get("source_refs", [])
                        if isinstance(ref, dict)
                    )
                )
            quality_artifact = await self.artifact_store.write_json(
                run_id=child_run_id,
                step_key="search_quality",
                source="filter",
                payload={
                    "effective_count": len(new_papers) + len(existing_papers),
                    "source_stats": source_stats,
                },
            )

            manifest = {
                "run_id": child_run_id,
                "step_key": "normalize_deduplicate_filter",
                "paper_info_schema": "paper_info_v1",
                "papers": new_papers,
                "rejected": rejected,
                "identity_conflicts": identity_conflicts,
                "deferred_filters": {
                    "requires_code": understanding.requiresCode,
                    "paper_tags": [item.value for item in search_tag.paperTag],
                },
            }

            artifact = await self.artifact_store.write_json(
                run_id=child_run_id,
                step_key="normalize_deduplicate_filter",
                source="normalized",
                payload=manifest,
            )

        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"去重过滤失败：{exc}",
            }

        return {
            "stage": "reviewing_search",
            "status": "running",
            "normalized_manifest_artifact_ref": artifact.artifact_uri,
            "existing_papers_manifest_artifact_ref": (existing_artifact.artifact_uri),
            "search_quality_artifact_ref": quality_artifact.artifact_uri,
            "provider_stats": source_stats,
            "progress": {
                **state.get("progress", {}),
                "deduplicated": len(normalized_papers),
                "accepted": len(normalized_papers),
                "existing_in_database": len(existing_papers),
                "new_candidates": len(new_papers),
                "filtered": filtered_count,
                "rejected": len(rejected),
            },
            "warnings": [
                *state.get("warnings", []),
                *(
                    ["requiresCode 需要代码链接补全节点验证，当前未作为过滤条件执行。"]
                    if understanding.requiresCode
                    else []
                ),
            ],
            "error": None,
        }
