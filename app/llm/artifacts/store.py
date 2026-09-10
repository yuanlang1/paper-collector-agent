from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from app.config import settings
from app.llm.artifacts.schemas import ArtifactRef


class ArtifactUriError(ValueError):
    """The URI does not identify a JSON artifact inside the local store."""


def _safe_path_part(value: str | None, default: str = "unknown") -> str:
    if not value:
        return default

    value = value.strip()
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    value = value.strip("._-")

    return value[:120] or default


def _short_hash(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


class LocalArtifactStore:
    """
    本地 artifact 存储。

    当前阶段保存到本地 JSON 文件。
    后续如果切换到 OSS / MinIO / S3，只需要替换这个类。
    """

    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(
            base_dir or getattr(settings, "ARTIFACT_BASE_DIR", "data/artifacts")
        ).resolve()

    def build_json_path(
        self,
        *,
        run_id: str,
        step_key: str,
        source: str,
        name_hint: str | None = None,
    ) -> Path:
        safe_run_id = _safe_path_part(run_id, "run")
        safe_step_key = _safe_path_part(step_key, "step")
        safe_source = _safe_path_part(source.lower(), "source")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = _short_hash(name_hint or f"{run_id}:{step_key}:{source}:{timestamp}")

        filename = f"{safe_source}_{timestamp}_{suffix}.json"

        return self.base_dir / safe_run_id / safe_step_key / filename

    def to_artifact_uri(self, path: Path) -> str:
        resolved = path.resolve()

        try:
            relative_path = resolved.relative_to(self.base_dir)
            return f"artifact://{relative_path.as_posix()}"
        except ValueError:
            return resolved.as_uri()

    def resolve_json_uri(self, artifact_uri: str) -> tuple[str, Path]:
        """Resolve an internal artifact URI without allowing directory traversal."""
        parsed = urlsplit(artifact_uri)
        if (
            parsed.scheme != "artifact"
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
        ):
            raise ArtifactUriError("invalid artifact URI")

        parts = [unquote(parsed.netloc)]
        parts.extend(unquote(part) for part in parsed.path.lstrip("/").split("/"))
        if len(parts) < 3 or any(
            not part
            or part in {".", ".."}
            or "/" in part
            or "\\" in part
            or ":" in part
            for part in parts
        ):
            raise ArtifactUriError("invalid artifact URI")

        path = (self.base_dir.joinpath(*parts)).resolve()
        try:
            path.relative_to(self.base_dir)
        except ValueError as exc:
            raise ArtifactUriError("artifact URI is outside the artifact store") from exc
        if path.suffix.lower() != ".json":
            raise ArtifactUriError("artifact is not a JSON file")
        return parts[0], path

    async def stage_run_directories(
        self,
        *,
        run_ids: tuple[str, ...],
        deletion_id: str,
    ) -> None:
        await asyncio.to_thread(
            self._stage_run_directories,
            run_ids,
            deletion_id,
        )

    async def purge_staged_directories(self, *, deletion_id: str) -> None:
        await asyncio.to_thread(self._purge_staged_directories, deletion_id)

    def _stage_run_directories(
        self,
        run_ids: tuple[str, ...],
        deletion_id: str,
    ) -> None:
        for run_id in run_ids:
            source = self._run_directory(run_id)
            target = self._staged_run_directory(deletion_id, run_id)
            if target.exists():
                if target.is_symlink() or not target.is_dir():
                    raise ArtifactUriError("invalid staged artifact directory")
                continue
            if not source.exists():
                continue
            if source.is_symlink() or not source.is_dir():
                raise ArtifactUriError("invalid artifact directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)

    def _purge_staged_directories(self, deletion_id: str) -> None:
        staged_directory = self._staged_directory(deletion_id)
        if not staged_directory.exists():
            return
        if staged_directory.is_symlink() or not staged_directory.is_dir():
            raise ArtifactUriError("invalid staged artifact directory")
        shutil.rmtree(staged_directory)

    def _run_directory(self, run_id: str) -> Path:
        return self._safe_directory(self.base_dir / self._path_part(run_id), run_id)

    def _staged_directory(self, deletion_id: str) -> Path:
        return self._safe_directory(
            self.base_dir / ".trash" / self._path_part(deletion_id),
            deletion_id,
        )

    def _staged_run_directory(self, deletion_id: str, run_id: str) -> Path:
        return self._safe_directory(
            self.base_dir
            / ".trash"
            / self._path_part(deletion_id)
            / self._path_part(run_id),
            run_id,
        )

    def _safe_directory(self, raw_path: Path, value: str) -> Path:
        if raw_path.is_symlink():
            raise ArtifactUriError(f"invalid artifact directory: {value}")
        path = raw_path.resolve()
        try:
            path.relative_to(self.base_dir)
        except ValueError as exc:
            raise ArtifactUriError("artifact directory is outside the artifact store") from exc
        return path

    @staticmethod
    def _path_part(value: str) -> str:
        if _safe_path_part(value) != value:
            raise ArtifactUriError("invalid artifact directory name")
        return value

    def write_json_sync(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = path.with_suffix(path.suffix + ".tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(
                payload,
                f,
                ensure_ascii=False,
                indent=2,
            )

        tmp_path.replace(path)

    async def write_json(
        self,
        *,
        run_id: str,
        step_key: str,
        source: str,
        kind: str,
        payload: dict[str, Any],
        count: int = 0,
        name_hint: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        path = self.build_json_path(
            run_id=run_id,
            step_key=step_key,
            source=source,
            name_hint=name_hint,
        )

        await asyncio.to_thread(
            self.write_json_sync,
            path,
            payload,
        )

        artifact_id = f"artifact_{uuid.uuid4().hex[:16]}"

        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_uri=self.to_artifact_uri(path),
            path=str(path),
            kind=kind,
            source=source,
            run_id=run_id,
            step_key=step_key,
            count=count,
            metadata=metadata or {},
        )

    def read_json(self, artifact: ArtifactRef) -> dict[str, Any]:
        path = Path(artifact.path)

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    async def read_json_uri(
        self,
        artifact_uri: str,
    ) -> dict[str, Any]:
        _, path = self.resolve_json_uri(artifact_uri)

        return await asyncio.to_thread(
            lambda: json.loads(
                path.read_text(encoding="utf-8")
            )
        )
