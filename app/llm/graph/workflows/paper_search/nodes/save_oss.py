from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.config import settings
from app.infrastructure.oss import (
    AliyunOssObjectStore,
    OssObjectStore,
    build_pdf_object_name,
    normalize_pdf_sha256,
)
from app.llm.artifacts.store import LocalArtifactStore


logger = logging.getLogger(__name__)


class SavePersistedPdfsToOssNode:
    def __init__(
        self,
        *,
        artifact_store: LocalArtifactStore | None = None,
        object_store: OssObjectStore | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.object_store = object_store or AliyunOssObjectStore()

    def _local_pdf_path(self, paper: dict[str, Any]) -> tuple[Path, str] | None:
        pdf_download = paper.get("pdf_download")
        if not isinstance(pdf_download, dict):
            return None

        local_path = pdf_download.get("local_pdf_path")
        sha256 = pdf_download.get("sha256")
        if not isinstance(local_path, str) or not isinstance(sha256, str):
            return None

        base_dir = self.artifact_store.base_dir.resolve()
        try:
            path = Path(local_path).resolve()
            path.relative_to(base_dir)
        except (OSError, ValueError):
            return None
        if not path.is_file():
            return None

        return path, sha256

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        if not settings.OSS_ENABLED:
            return {}

        run_id = state.get("run_id")
        persisted_uri = state.get("persisted_papers_manifest_artifact_ref")
        recommendation_uri = state.get(
            "task_bound_recommendation_manifest_artifact_ref"
        ) or state.get("recommendation_manifest_artifact_ref")
        if (
            not isinstance(run_id, str)
            or not isinstance(persisted_uri, str)
            or not isinstance(recommendation_uri, str)
        ):
            return {}

        try:
            persisted = await self.artifact_store.read_json_uri(persisted_uri)
            recommendation = await self.artifact_store.read_json_uri(
                recommendation_uri
            )
        except Exception as exc:
            logger.warning("Skipping OSS PDF upload: unable to read manifest: %s", exc)
            return {}

        persisted_results = persisted.get("results")
        recommendation_papers = recommendation.get("papers")
        if not isinstance(persisted_results, list) or not isinstance(
            recommendation_papers, list
        ):
            return {}

        papers_by_client_key = {
            f"{run_id}:{index}": paper
            for index, paper in enumerate(recommendation_papers)
            if isinstance(paper, dict)
        }
        for result in persisted_results:
            if not isinstance(result, dict):
                continue
            if result.get("paper_save_status") != "saved":
                continue

            pdf_sha256 = normalize_pdf_sha256(result.get("oss_name"))
            paper = papers_by_client_key.get(result.get("client_key"))
            if pdf_sha256 is None or not isinstance(paper, dict):
                continue

            local_pdf = self._local_pdf_path(paper)
            if local_pdf is None:
                continue
            source_path, source_sha256 = local_pdf
            expected_pdf_sha256 = normalize_pdf_sha256(source_sha256)
            if pdf_sha256 != expected_pdf_sha256:
                continue
            object_name = build_pdf_object_name(
                prefix=settings.OSS_PREFIX,
                pdf_sha256=pdf_sha256,
            )
            if object_name is None:
                continue

            try:
                await self.object_store.upload_pdf(
                    source_path=source_path,
                    object_name=object_name,
                    sha256=pdf_sha256,
                )
            except Exception as exc:
                logger.warning(
                    "OSS PDF upload failed for paper_id=%s object_name=%s: %s",
                    result.get("paper_id"),
                    object_name,
                    exc,
                )

        return {}
