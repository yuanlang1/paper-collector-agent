from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Mapping
from typing import Any

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.tools.file_tools.download_file import download_file_handler


LIBRARY_SAVE_DIR = "papers"
PDF_DOWNLOAD_CONCURRENCY = 4


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
        child_run_id: str,
    ) -> dict[str, Any]:
        result = await download_file_handler(
            {
                "url": str(paper_info["pdf_url"]),
                "save_dir": (
                    f"{LIBRARY_SAVE_DIR}/"
                    f"{_safe_filename_part(child_run_id, max_length=120)}"
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

    async def _download_paper(
        self,
        paper: dict[str, Any],
        child_run_id: str,
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        paper_info = paper["paper_info"]
        title = str(paper_info.get("title") or "")
        try:
            file_name = self._pdf_file_name(paper_info)
            async with semaphore:
                downloaded = await self._download_one(paper_info, child_run_id)

            paper_info["file_name"] = file_name
            paper["pdf_download"] = {
                "status": "success",
                "file_name": file_name,
                "local_pdf_path": downloaded["saved_path"],
                "artifact_uri": downloaded["artifact_uri"],
                "final_url": downloaded["final_url"],
                "sha256": downloaded["sha256"],
                "size_bytes": downloaded["size_bytes"],
            }
            return {
                "paper": paper,
                "removed": None,
                "warning": None,
                "saved_path": downloaded["saved_path"],
            }
        except Exception as exc:
            reason = f"PDF 下载失败：{exc}"
            return {
                "paper": None,
                "removed": {"title": title, "reason": reason},
                "warning": f"{title} 的 {reason}",
                "saved_path": None,
            }

    @staticmethod
    async def _gather_or_cancel(
        tasks: list[asyncio.Task[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        try:
            return list(await asyncio.gather(*tasks))
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        downloaded_pdf_paths: list[str] = []

        try:
            child_run_id = state["child_run_id"]
            artifact_uri = state.get("venue_manifest_artifact_ref")

            manifest = await self.artifact_store.read_json_uri(artifact_uri)
            papers = manifest.get("papers", [])

            downloaded_papers: list[dict[str, Any]] = []
            removed_papers: list[dict[str, str]] = []
            warnings: list[str] = []

            semaphore = asyncio.Semaphore(PDF_DOWNLOAD_CONCURRENCY)
            tasks = [
                asyncio.create_task(
                    self._download_paper(paper, child_run_id, semaphore)
                )
                for paper in papers
            ]
            for result in await self._gather_or_cancel(tasks):
                paper = result["paper"]
                removed = result["removed"]
                warning = result["warning"]
                saved_path = result["saved_path"]
                if paper is not None:
                    downloaded_papers.append(paper)
                if removed is not None:
                    removed_papers.append(removed)
                if warning is not None:
                    warnings.append(warning)
                if saved_path is not None:
                    downloaded_pdf_paths.append(saved_path)

            manifest["papers"] = downloaded_papers
            manifest["removed_pdf_download_failed"] = removed_papers
            manifest["step_key"] = "download_pdfs"

            artifact = await self.artifact_store.write_json(
                run_id=child_run_id,
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

            has_existing = False
            if not downloaded_papers and state.get("existing_papers_manifest_artifact_ref"):
                existing_manifest = await self.artifact_store.read_json_uri(
                    state["existing_papers_manifest_artifact_ref"]
                )
                has_existing = bool(existing_manifest.get("papers"))
                if has_existing:
                    warnings.append("无新增 PDF 可用，继续处理已有论文。")

        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "downloaded_pdf_paths": downloaded_pdf_paths,
                "error": f"PDF 下载失败：{exc}",
            }

        can_continue = bool(downloaded_papers) or has_existing
        if not can_continue:
            warnings.append("所有论文的 PDF 下载均失败。" if papers else "没有可继续处理的论文。")
        return {
            "stage": "enriching_abstract" if can_continue else "failed",
            "status": "running" if can_continue else "failed",
            "pdf_manifest_artifact_ref": artifact.artifact_uri,
            "downloaded_pdf_paths": downloaded_pdf_paths,
            "progress": {
                **state.get("progress", {}),
                "pdf_downloaded": len(downloaded_papers),
                "pdf_download_failed": len(removed_papers),
            },
            "warnings": [*state.get("warnings", []), *warnings],
            "error": None if can_continue else "没有可继续处理的论文。",
        }
