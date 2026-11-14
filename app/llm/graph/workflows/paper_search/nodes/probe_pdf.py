from collections.abc import Mapping
from typing import Any

from anyio import fail_after
from langgraph.errors import GraphBubbleUp
from langgraph.types import Overwrite

from app.config import settings
from app.llm.artifacts.schemas import ArtifactRef
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.paper_search.paper_search_schemas import PaperTypeCode
from app.tasks.modules.paper_search.crossref_enrich import (
    normalize_doi,
    optional_text,
    search_crossref_paper,
)
from app.utils.pdf_probe import probe_pdf_url


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


def _merge_crossref_metadata(
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
    paper_type_code = _paper_type_code(crossref_paper.get("publication_type"))

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
        "crossref_type": optional_text(crossref_paper.get("publication_type")),
    }


class PdfProbeNode:
    def __init__(self, artifact_store: LocalArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        try:
            # ponytail: read one local candidate manifest in place, without an offload.
            manifest = self.artifact_store.read_json(
                ArtifactRef(artifact_uri=state["normalized_manifest_artifact_ref"])
            )
            candidates = []
            warnings = list(state.get("warnings", []))
            missing = failed = filled = 0

            for paper in manifest["papers"]:
                info = paper["paper_info"]
                url = info.get("pdf_url")
                if not url:
                    try:
                        with fail_after(settings.PAPER_CROSSREF_TASK_TIMEOUT_SECONDS):
                            match, method = await search_crossref_paper(
                                {
                                    **info,
                                    "authors": paper.get("author_names")
                                    or info.get("authors"),
                                }
                            )
                    except GraphBubbleUp:
                        raise
                    except Exception:
                        missing += 1
                        warnings.append(
                            f"{info['title']} 的 Crossref PDF 链接查询失败。"
                        )
                        continue
                    url = optional_text(match.get("pdf_url")) if match else None
                    if not url:
                        missing += 1
                        continue
                    info["pdf_url"] = url
                    _merge_crossref_metadata(paper, match, method)
                    filled += 1

                if await probe_pdf_url(url):
                    candidates.append(paper)
                else:
                    failed += 1

        except GraphBubbleUp:
            raise
        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"PDF 探测失败：{exc}",
            }

        if missing:
            warnings.append(f"{missing} 篇新增论文没有 PDF 链接。")
        if failed:
            warnings.append(f"{failed} 篇新增论文的 PDF 链接未通过探测。")
        return {
            "enrich_candidates": candidates,
            "enrich_results": Overwrite({}),
            "progress": {
                **state.get("progress", {}),
                "removed_without_pdf": missing,
                "crossref_pdf_filled": filled,
                "pdf_probe_passed": len(candidates),
                "pdf_probe_failed": failed,
            },
            "warnings": warnings,
        }
