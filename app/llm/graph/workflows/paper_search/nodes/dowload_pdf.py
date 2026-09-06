from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.tools.file_tools.download_file import DOWNLOAD_FILE_TOOL


LIBRARY_SAVE_DIR = "papers"


def _safe_filename_part(value: str, max_length: int) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value.strip())
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ._")[:max_length] or "unknown"


def _normalize_text(value: Any) -> str:
    return re.sub(r"\W+", "", str(value or "").casefold())


def _paper_identity(paper_info: Mapping[str, Any]) -> str:
    doi = str(paper_info.get("doi") or "").strip().casefold()

    if doi:
        return f"doi:{doi}"

    return (
        f"title:{_normalize_text(paper_info.get('title'))}"
        f"|authors:{_normalize_text(paper_info.get('authors'))}"
    )


def _first_author(authors: Any) -> str:
    parts = [
        item.strip()
        for item in re.split(r"[,，;；]", str(authors or ""))
        if item.strip()
    ]
    return parts[0] if parts else "unknown_author"

class PdfDownloadNode:
    def __init__(
        self,
        artifact_store: LocalArtifactStore | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()

    def _pdf_file_name(
        self,
        paper_info: Mapping[str, Any],
    ) -> str:
        title = _safe_filename_part(
            str(paper_info.get("title") or "unknown_paper"),
            max_length=120,
        )
        author = _safe_filename_part(
            _first_author(paper_info.get("authors")),
            max_length=60,
        )
        identity_hash = hashlib.sha256(
            _paper_identity(paper_info).encode("utf-8")
        ).hexdigest()[:12]
        return f"{title}_{author}_{identity_hash}.pdf"

    async def _download_one(
        self,
        paper_info: dict[str, Any],
        run_id: str,
    ) -> dict[str, Any]:
        result = await DOWNLOAD_FILE_TOOL.fn(
            {
                "url": str(paper_info["pdf_url"]),
                "save_dir": (
                    f"{LIBRARY_SAVE_DIR}/"
                    f"{_safe_filename_part(run_id, max_length=120)}"
                ),
                "file_name": self._pdf_file_name(paper_info),
                "expected_type": "pdf",
                "overwrite": False,
            },
            None,
        )

        if result.get("ok"):
            return result["data"]

        error = result.get("error") or {}
        raise ValueError(error.get("message") or "PDF 下载失败。")

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        downloaded_pdf_paths: list[str] = []

        try:
            run_id = state.get("run_id")
            artifact_uri = state.get("venue_manifest_artifact_ref")

            if not isinstance(run_id, str) or not run_id:
                raise ValueError("缺少有效 run_id。")

            if (
                not isinstance(artifact_uri, str)
                or not artifact_uri.startswith("artifact://")
            ):
                raise ValueError("缺少 venue manifest artifact。")

            artifact_base_dir = self.artifact_store.base_dir.resolve()
            manifest_path = (
                artifact_base_dir
                / artifact_uri.removeprefix("artifact://")
            ).resolve()

            try:
                manifest_path.relative_to(artifact_base_dir)
            except ValueError as exc:
                raise ValueError(
                    "PDF manifest 超出 artifact 存储目录。"
                ) from exc

            def read_manifest() -> dict[str, Any]:
                with manifest_path.open("r", encoding="utf-8") as file:
                    return json.load(file)

            manifest = await asyncio.to_thread(read_manifest)
            papers = manifest.get("papers", [])

            if not isinstance(papers, list):
                raise ValueError("manifest papers 格式无效。")

            downloaded_papers: list[dict[str, Any]] = []
            removed_papers: list[dict[str, str]] = []
            warnings: list[str] = []

            for paper in papers:
                paper_info = paper.get("paper_info")

                if (
                    not isinstance(paper_info, dict)
                    or not paper_info.get("pdf_url")
                ):
                    removed_papers.append(
                        {
                            "title": str(
                                paper_info.get("title")
                                if isinstance(paper_info, dict)
                                else ""
                            ),
                            "reason": "缺少 PDF 下载地址。",
                        }
                    )
                    continue

                try:
                    file_name = self._pdf_file_name(paper_info)
                    downloaded = await self._download_one(paper_info, run_id)
                    downloaded_pdf_paths.append(downloaded["saved_path"])

                    paper_info["file_name"] = file_name
                    pdf_download = {
                        "status": "success",
                        "file_name": file_name,
                        "local_pdf_path": downloaded["saved_path"],
                        "artifact_uri": downloaded["artifact_uri"],
                        "final_url": downloaded["final_url"],
                        "sha256": downloaded["sha256"],
                        "size_bytes": downloaded["size_bytes"],
                    }
                    paper["pdf_download"] = pdf_download

                    downloaded_papers.append(paper)

                except Exception as exc:
                    title = str(paper_info.get("title") or "")

                    removed_papers.append(
                        {
                            "title": title,
                            "reason": f"PDF 下载失败：{exc}",
                        }
                    )
                    warnings.append(
                        f"{title} 的 PDF 下载失败：{exc}"
                    )

            manifest["papers"] = downloaded_papers
            manifest["removed_pdf_download_failed"] = removed_papers
            manifest["step_key"] = "download_pdfs"

            artifact = await self.artifact_store.write_json(
                run_id=run_id,
                step_key="download_pdfs",
                source="pdf",
                kind="paper_info_pdf_manifest_json",
                payload=manifest,
                count=len(downloaded_papers),
                metadata={
                    "input_manifest": artifact_uri,
                    "downloaded_count": len(downloaded_papers),
                    "removed_count": len(removed_papers),
                },
            )

        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "downloaded_pdf_paths": downloaded_pdf_paths,
                "error": f"PDF 下载失败：{exc}",
            }

        if not downloaded_papers:
            return {
                "stage": "failed",
                "status": "failed",
                "pdf_manifest_artifact_ref": artifact.artifact_uri,
                "downloaded_pdf_paths": downloaded_pdf_paths,
                "warnings": [
                    *state.get("warnings", []),
                    *warnings,
                    "所有论文的 PDF 下载均失败。",
                ],
                "error": "没有可继续处理的论文。",
            }

        return {
            "stage": "enriching_abstract",
            "status": (
                "partial_failed"
                if removed_papers
                else "running"
            ),
            "degraded": (
                bool(state.get("degraded"))
                or bool(removed_papers)
            ),
            "pdf_manifest_artifact_ref": artifact.artifact_uri,
            "downloaded_pdf_paths": downloaded_pdf_paths,
            "progress": {
                **state.get("progress", {}),
                "pdf_downloaded": len(downloaded_papers),
                "pdf_download_failed": len(removed_papers),
            },
            "warnings": [
                *state.get("warnings", []),
                *warnings,
            ],
            "error": None,
        }
