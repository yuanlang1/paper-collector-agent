from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.llm.artifacts.store import LocalArtifactStore


class PaperEnrichmentNode:
    """Prepare Google Scholar results for metadata and PDF enrichment."""

    def __init__(
        self,
        artifact_store: LocalArtifactStore | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            child_run_id = state["child_run_id"]
            artifact_uri = state.get("normalized_manifest_artifact_ref")

            if (
                not isinstance(artifact_uri, str)
                or not artifact_uri.startswith("artifact://")
            ):
                raise ValueError("缺少 normalized manifest artifact。")

            base_dir = self.artifact_store.base_dir.resolve()
            path = (
                base_dir / artifact_uri.removeprefix("artifact://")
            ).resolve()
            try:
                path.relative_to(base_dir)
            except ValueError as exc:
                raise ValueError("manifest artifact 超出存储目录。") from exc

            def read_manifest() -> dict[str, Any]:
                with path.open("r", encoding="utf-8") as file:
                    return json.load(file)

            manifest = await asyncio.to_thread(read_manifest)
            papers = manifest.get("papers", [])
            if not isinstance(papers, list):
                raise ValueError("manifest papers 格式无效。")

            kept_papers: list[dict[str, Any]] = []
            removed_papers: list[dict[str, str]] = []

            for paper in papers:
                if not isinstance(paper, dict):
                    continue
                paper_info = paper.get("paper_info")
                if not isinstance(paper_info, dict):
                    continue
                if not paper_info.get("pdf_url"):
                    removed_papers.append(
                        {
                            "title": str(paper_info.get("title") or ""),
                            "reason": "未找到可用 PDF 下载地址。",
                        },
                    )
                    continue
                kept_papers.append(paper)

            manifest["papers"] = kept_papers
            manifest["removed_without_pdf"] = removed_papers
            manifest["step_key"] = "paper_enrichment"
            artifact = await self.artifact_store.write_json(
                run_id=child_run_id,
                step_key="paper_enrichment",
                source="enriched",
                kind="paper_info_enriched_manifest_json",
                payload=manifest,
                count=len(kept_papers),
                metadata={
                    "input_manifest": artifact_uri,
                    "pdf_enriched_count": 0,
                    "venue_enriched_count": 0,
                    "removed_without_pdf_count": len(removed_papers),
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
                if state.get("degraded") or removed_papers
                else "running"
            ),
            "degraded": bool(state.get("degraded")) or bool(removed_papers),
            "enrichment_manifest_artifact_ref": artifact.artifact_uri,
            "progress": {
                **state.get("progress", {}),
                "pdf_enriched": 0,
                "venue_enriched": 0,
                "removed_without_pdf": len(removed_papers),
            },
            "warnings": list(state.get("warnings", [])),
            "error": None,
        }
