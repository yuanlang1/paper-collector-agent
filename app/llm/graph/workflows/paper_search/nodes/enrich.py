from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.paper_search_schemas import PaperTypeCode
from app.tasks.modules.paper_search.crossref_enrich import (
    normalize_doi,
    optional_text,
    query_papers,
)


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


def _append_venue_candidate(
    paper: dict[str, Any],
    venue: str | None,
) -> None:
    if venue is None:
        return

    candidates = paper.setdefault("venue_candidates", [])
    if venue not in candidates:
        candidates.append(venue)


class PaperEnrichmentNode:
    """Filter PDF candidates and supplement their metadata with Crossref."""

    def __init__(self, artifact_store: LocalArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()

    def _merge(
        self,
        paper: dict[str, Any],
        crossref_paper: Mapping[str, Any],
        match_method: str,
    ) -> None:
        paper_info = paper["paper_info"]
        doi = normalize_doi(crossref_paper.get("doi"))
        published_date = optional_text(crossref_paper.get("published"))
        venue_title = optional_text(crossref_paper.get("venue"))
        citations = max(
            _as_int(paper_info.get("citations")),
            _as_int(crossref_paper.get("citation_count")),
        )
        paper_type_code = _paper_type_code(
            crossref_paper.get("publication_type")
        )

        if not paper_info.get("doi") and doi:
            paper_info["doi"] = doi
        if published_date and (
            not paper_info.get("publish_date")
            or paper_info.get("publish_date_precision") == "year"
        ):
            paper_info["publish_date"] = published_date
            paper_info["publish_date_precision"] = "day"
        abstract = optional_text(crossref_paper.get("abstract"))
        if abstract and (
            not paper_info.get("paper_abstract")
            or paper_info.get("abstract_source") == "search_snippet"
        ):
            paper_info["paper_abstract"] = abstract
            paper_info["abstract_source"] = "crossref"
        if not paper_info.get("abstract_url"):
            paper_info["abstract_url"] = optional_text(
                crossref_paper.get("abstract_url")
            ) or optional_text(crossref_paper.get("landing_url"))
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
            "crossref_type": optional_text(
                crossref_paper.get("publication_type")
            ),
        }

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        try:
            manifest = await self.artifact_store.read_json_uri(state["normalized_manifest_artifact_ref"])
            papers = []
            removed = []
            for paper in manifest.get("papers", []):
                info = paper["paper_info"]
                if info.get("pdf_url"):
                    papers.append(paper)
                else:
                    removed.append({
                        "title": str(info.get("title") or ""),
                        "reason": "未找到可用 PDF 下载地址。",
                    })

            matched = unmatched = 0
            warnings = []
            results = await query_papers([paper["paper_info"] for paper in papers])
            for index, paper in enumerate(papers):
                result = results[index]
                if "error" in result:
                    paper["crossref_metadata"] = {"status": "failed", "match_method": None}
                    title = optional_text(paper["paper_info"].get("title")) or ""
                    warnings.append(f"{title} 的 Crossref 补充失败：{result['error']}")
                elif result["candidate"] is None:
                    paper["crossref_metadata"] = {
                        "status": "not_matched",
                        "match_method": result["match_method"],
                    }
                    unmatched += 1
                else:
                    self._merge(paper, result["candidate"], result["match_method"])
                    matched += 1

            manifest["papers"] = papers
            manifest["removed_without_pdf"] = removed
            manifest["step_key"] = "crossref_enrichment"
            artifact = await self.artifact_store.write_json(
                run_id=state["child_run_id"],
                step_key="crossref_enrichment",
                source="crossref",
                payload=manifest,
            )
        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"Crossref metadata enrichment failed: {exc}",
            }

        return {
            "stage": "resolving_venues",
            "status": "running",
            "crossref_enrichment_manifest_artifact_ref": artifact.artifact_uri,
            "progress": {
                **state.get("progress", {}),
                "pdf_enriched": 0,
                "venue_enriched": 0,
                "removed_without_pdf": len(removed),
                "crossref_matched": matched,
                "crossref_unmatched": unmatched,
                "crossref_failed": len(warnings),
            },
            "warnings": [*state.get("warnings", []), *warnings],
            "error": None,
        }
