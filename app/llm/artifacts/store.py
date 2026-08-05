from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.llm.artifacts.schemas import ArtifactRef


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
        path = self.base_dir / artifact_uri.removeprefix("artifact://")

        return await asyncio.to_thread(
            lambda: json.loads(
                path.read_text(encoding="utf-8")
            )
        )
