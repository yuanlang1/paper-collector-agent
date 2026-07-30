from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.tools.search_tools.arxiv.search_args import (
    ArxivSearchArgs,
)
from app.llm.tools.search_tools.arxiv.search_arxiv import (
    arxiv_search_handler,
)
from app.llm.tools.search_tools.dblp.search_args import (
    DblpSearchArgs,
)
from app.llm.tools.search_tools.dblp.search_dblp import (
    dblp_search_handler,
)


def _text(value: Any) -> str | None:
    if value is None:
        return None

    value = str(value).strip()
    return value or None


def _normalize_title(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s]", "", value)
    return value.strip()


def _source_name(value: Any) -> str:
    value = (_text(value) or "").casefold()

    mapping = {
        "arxiv": "arXiv",
        "dblp": "DBLP",
        "google_scholar": "Google Scholar",
        "google scholar": "Google Scholar",
    }

    return mapping.get(value, _text(value) or "Unknown")


def _paper_sources(paper: dict[str, Any]) -> set[str]:
    sources = {
        _source_name(paper.get("paper_info", {}).get("source")),
    }

    for source_ref in paper.get("source_refs", []):
        if isinstance(source_ref, dict):
            sources.add(_source_name(source_ref.get("source")))

    return sources


def _find_exact_title_match(
    title: str,
    papers: list[dict[str, Any]],
) -> dict[str, Any] | None:
    target = _normalize_title(title)

    for paper in papers:
        candidate_title = _text(paper.get("title"))

        if (
            candidate_title
            and _normalize_title(candidate_title) == target
        ):
            return paper

    return None


def _append_source_ref(
    paper: dict[str, Any],
    *,
    source: str,
    source_paper: dict[str, Any],
) -> None:
    source_refs = paper.setdefault("source_refs", [])

    reference = {
        "source": source,
        "source_id": source_paper.get("source_id"),
        "doi": source_paper.get("doi"),
        "arxiv_id": source_paper.get("arxiv_id"),
        "landing_url": (
            source_paper.get("landing_url")
            or source_paper.get("abstract_url")
        ),
    }

    if reference not in source_refs:
        source_refs.append(reference)


class PaperEnrichmentNode:
    def __init__(
        self,
        artifact_store: LocalArtifactStore | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()

    async def _enrich_pdf_from_arxiv(
        self,
        paper: dict[str, Any],
    ) -> str | None:
        paper_info = paper["paper_info"]
        title = paper_info["title"]

        arxiv_id = next(
            (
                source_ref.get("arxiv_id")
                for source_ref in paper.get("source_refs", [])
                if isinstance(source_ref, dict)
                and source_ref.get("arxiv_id")
            ),
            None,
        )

        args = (
            ArxivSearchArgs(
                arxiv_ids=[arxiv_id],
                include_abstract=True,
                max_results=1,
                total_limit=1,
            )
            if arxiv_id
            else ArxivSearchArgs(
                query=title,
                search_type="title",
                include_abstract=True,
                max_results=3,
                total_limit=3,
            )
        )

        result = await arxiv_search_handler(
            args.model_dump(mode="json"),
            None,
        )

        if result.get("ok") is not True:
            return None

        matched = (
            result["papers"][0]
            if arxiv_id and result.get("papers")
            else _find_exact_title_match(
                title,
                result.get("papers", []),
            )
        )

        if not matched:
            return None

        pdf_url = _text(matched.get("pdf_url"))

        if not pdf_url:
            return None

        paper_info["pdf_url"] = pdf_url

        if not paper_info.get("abstract_url"):
            paper_info["abstract_url"] = _text(
                matched.get("abstract_url")
            )

        if (
            not paper_info.get("paper_abstract")
            and matched.get("abstract")
        ):
            paper_info["paper_abstract"] = matched["abstract"]

        paper.setdefault("pdf_candidates", []).append(pdf_url)
        _append_source_ref(
            paper,
            source="arXiv",
            source_paper=matched,
        )

        return pdf_url

    async def _enrich_venue_from_dblp(
        self,
        paper: dict[str, Any],
    ) -> str | None:
        paper_info = paper["paper_info"]
        title = paper_info["title"]

        args = DblpSearchArgs(
            query=title,
            h=5,
            total_limit=5,
        )

        result = await dblp_search_handler(
            args.model_dump(mode="json"),
            None,
        )

        if result.get("ok") is not True:
            return None

        matched = _find_exact_title_match(
            title,
            result.get("papers", []),
        )

        if not matched:
            return None

        venue = _text(matched.get("venue"))

        if not venue:
            return None

        venue_candidates = paper.setdefault(
            "venue_candidates",
            [],
        )

        if venue not in venue_candidates:
            venue_candidates.append(venue)

        if not paper_info.get("abstract_url"):
            paper_info["abstract_url"] = _text(
                matched.get("landing_url")
            )

        _append_source_ref(
            paper,
            source="DBLP",
            source_paper=matched,
        )

        return venue

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            run_id = state.get("run_id")
            artifact_uri = state.get(
                "normalized_manifest_artifact_ref"
            )

            if not isinstance(run_id, str) or not run_id:
                raise ValueError("缺少有效 run_id。")

            if (
                not isinstance(artifact_uri, str)
                or not artifact_uri.startswith("artifact://")
            ):
                raise ValueError(
                    "缺少 normalized manifest artifact。"
                )

            base_dir = self.artifact_store.base_dir.resolve()
            path = (
                base_dir
                / artifact_uri.removeprefix("artifact://")
            ).resolve()

            try:
                path.relative_to(base_dir)
            except ValueError as exc:
                raise ValueError(
                    "manifest artifact 超出存储目录。"
                ) from exc

            def read_manifest() -> dict[str, Any]:
                with path.open("r", encoding="utf-8") as file:
                    return json.load(file)

            manifest = await asyncio.to_thread(read_manifest)
            papers = manifest.get("papers", [])

            if not isinstance(papers, list):
                raise ValueError("manifest papers 格式无效。")

            kept_papers: list[dict[str, Any]] = []
            removed_papers: list[dict[str, str]] = []
            warnings: list[str] = []

            pdf_enriched_count = 0
            venue_enriched_count = 0

            for paper in papers:
                if not isinstance(paper, dict):
                    continue

                paper_info = paper.get("paper_info")

                if not isinstance(paper_info, dict):
                    continue

                sources = _paper_sources(paper)

                # PDF 缺失，且原检索来源不是 arXiv 时才补充。
                if (
                    not paper_info.get("pdf_url")
                    and "arXiv" not in sources
                ):
                    try:
                        if await self._enrich_pdf_from_arxiv(paper):
                            pdf_enriched_count += 1
                    except Exception as exc:
                        warnings.append(
                            f"{paper_info.get('title')} 的 arXiv PDF "
                            f"补充失败：{exc}"
                        )

                # venue 缺失，且原检索来源不是 DBLP 时才补充。
                if (
                    not paper.get("venue_candidates")
                    and "DBLP" not in sources
                ):
                    try:
                        if await self._enrich_venue_from_dblp(paper):
                            venue_enriched_count += 1
                    except Exception as exc:
                        warnings.append(
                            f"{paper_info.get('title')} 的 DBLP venue "
                            f"补充失败：{exc}"
                        )

                # PDF 是后续流程的前置条件；未补齐则移除。
                if not paper_info.get("pdf_url"):
                    removed_papers.append(
                        {
                            "title": str(
                                paper_info.get("title") or ""
                            ),
                            "reason": "未找到可用 PDF 下载地址。",
                        }
                    )
                    continue

                kept_papers.append(paper)

            manifest["papers"] = kept_papers
            manifest["removed_without_pdf"] = removed_papers
            manifest["step_key"] = "paper_enrichment"

            artifact = await self.artifact_store.write_json(
                run_id=run_id,
                step_key="paper_enrichment",
                source="enriched",
                kind="paper_info_enriched_manifest_json",
                payload=manifest,
                count=len(kept_papers),
                metadata={
                    "input_manifest": artifact_uri,
                    "pdf_enriched_count": pdf_enriched_count,
                    "venue_enriched_count": venue_enriched_count,
                    "removed_without_pdf_count": len(
                        removed_papers
                    ),
                },
            )

        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"论文信息补充失败：{exc}",
            }

        return {
            "stage": "enriching_crossref",
            "status": (
                "partial_failed"
                if state.get("degraded") or warnings or removed_papers
                else "running"
            ),
            "degraded": (
                bool(state.get("degraded"))
                or bool(warnings)
                or bool(removed_papers)
            ),
            "enrichment_manifest_artifact_ref": (
                artifact.artifact_uri
            ),
            "progress": {
                **state.get("progress", {}),
                "pdf_enriched": pdf_enriched_count,
                "venue_enriched": venue_enriched_count,
                "removed_without_pdf": len(removed_papers),
            },
            "warnings": [
                *state.get("warnings", []),
                *warnings,
            ],
            "error": None,
        }
