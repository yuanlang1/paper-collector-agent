from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.llm.artifacts.store import LocalArtifactStore


class CleanupDownloadedPdfsNode:
    def __init__(
        self,
        artifact_store: LocalArtifactStore | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()

    def _delete(self, paths: list[str]) -> int:
        base_dir = self.artifact_store.base_dir.resolve()
        deleted = 0

        for value in paths:
            path = Path(value).resolve()
            path.relative_to(base_dir)
            if path.exists():
                path.unlink()
                deleted += 1

        return deleted

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        paths = state.get("downloaded_pdf_paths") or []
        try:
            deleted = await asyncio.to_thread(self._delete, paths)
        except (OSError, ValueError) as exc:
            stage = state.get("stage")
            status = state.get("status")
            if stage == "completed":
                stage = "partial_failed"
            if status == "completed":
                status = "partial_failed"
            return {
                "stage": stage,
                "status": status,
                "degraded": True,
                "pdf_cleanup_error": str(exc),
                "progress": {
                    **state.get("progress", {}),
                    "pdf_files_deleted": 0,
                },
                "warnings": [
                    *state.get("warnings", []),
                    f"PDF cleanup failed: {exc}",
                ],
                "error": (
                    state.get("error")
                    or f"PDF cleanup failed: {exc}"
                ),
            }

        return {
            "downloaded_pdf_paths": [],
            "pdf_cleanup_error": None,
            "progress": {
                **state.get("progress", {}),
                "pdf_files_deleted": deleted,
            },
        }
