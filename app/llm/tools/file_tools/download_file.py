from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import time
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Dict, Literal
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.tools.registry import Tool, ToolExecutionContext
from app.llm.tools.search_tools.common import (
    RETRYABLE_STATUS_CODES,
    elapsed_ms,
)

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
MAX_REDIRECTS = 3
MAX_DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT_SECONDS = 30.0
CHUNK_SIZE = 64 * 1024


class DownloadFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(
        ...,
        min_length=1,
        max_length=2_000,
        description="待下载文件的 HTTP 或 HTTPS 链接。",
    )
    save_dir: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="相对于 ARTIFACT_BASE_DIR 的保存目录。",
    )
    file_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="保存后的文件名。",
    )
    expected_type: Literal["pdf", "any"] = Field(
        "pdf",
        description="pdf 会校验 PDF 文件头；any 不限制文件类型。",
    )
    overwrite: bool = Field(False, description="是否覆盖同名文件。")

    @field_validator("save_dir")
    @classmethod
    def validate_save_dir(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        path = PurePosixPath(normalized)
        if (
            not normalized
            or path.is_absolute()
            or normalized == "."
            or any(part in {".", ".."} for part in path.parts)
            or ":" in normalized
        ):
            raise ValueError("save_dir 必须是安全的相对目录。")
        return normalized

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, value: str) -> str:
        name = value.strip()
        if (
            not name
            or "/" in name
            or "\\" in name
            or ".." in name
            or ":" in name
        ):
            raise ValueError("file_name 只能是文件名，不能包含路径。")
        return name


def _failure(
    *,
    args: DownloadFileArgs,
    started_at: float,
    code: str,
    message: str,
    retryable: bool = False,
    status_code: int | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "duration_ms": elapsed_ms(started_at),
    }

    if status_code is not None:
        metadata["status_code"] = status_code

    return {
        "ok": False,
        "url": args.url,
        "save_dir": args.save_dir,
        "file_name": args.file_name,
        "data": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
        "metadata": metadata,
    }


def _target_path(
    *,
    artifact_store: LocalArtifactStore,
    save_dir: str,
    file_name: str,
) -> Path:
    base_dir = artifact_store.base_dir.resolve()
    target = (base_dir / save_dir / file_name).resolve()

    try:
        target.relative_to(base_dir)
    except ValueError as exc:
        raise ValueError("保存路径超出 artifact 存储目录。") from exc

    return target


def _ensure_extension(args: DownloadFileArgs) -> str:
    if args.expected_type != "pdf":
        return args.file_name

    if args.file_name.lower().endswith(".pdf"):
        return args.file_name

    return f"{args.file_name}.pdf"


def _is_pdf(
    *,
    content_type: str | None,
    prefix: bytes,
) -> bool:
    normalized_content_type = (
        (content_type or "").split(";")[0].strip().lower()
    )

    return (
        prefix.startswith(b"%PDF-")
        and normalized_content_type in {
            "",
            "application/pdf",
            "application/octet-stream",
        }
    )


def _inspect_existing_file(
    *,
    target: Path,
    expected_type: str,
) -> dict[str, Any]:
    size_bytes = 0
    sha256 = hashlib.sha256()
    prefix = b""

    with target.open("rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            size_bytes += len(chunk)

            if size_bytes > MAX_FILE_SIZE_BYTES:
                raise ValueError("文件大小超过 50 MB 限制。")

            if len(prefix) < 8:
                prefix += chunk[: 8 - len(prefix)]

            sha256.update(chunk)

    if expected_type == "pdf" and not _is_pdf(
        content_type=None,
        prefix=prefix,
    ):
        raise ValueError("已有文件不是有效的 PDF 文件。")

    return {
        "size_bytes": size_bytes,
        "sha256": sha256.hexdigest(),
    }


async def _download(
    *,
    url: str,
    target: Path,
    expected_type: str,
    overwrite: bool,
) -> dict[str, Any]:
    current_url = url
    redirect_count = 0

    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not overwrite:
        raise FileExistsError("目标文件已存在。")

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(DOWNLOAD_TIMEOUT_SECONDS),
        follow_redirects=False,
        headers={
            "User-Agent": "paper-collector-agent/1.0",
            "Accept": "application/pdf,application/octet-stream,*/*",
        },
    ) as client:
        while True:
            async with client.stream("GET", current_url) as response:
                if response.status_code in {
                    301,
                    302,
                    303,
                    307,
                    308,
                }:
                    location = response.headers.get("location")

                    if not location:
                        raise ValueError("下载重定向缺少 Location 响应头。")

                    if redirect_count >= MAX_REDIRECTS:
                        raise ValueError("下载重定向次数超过限制。")

                    current_url = urljoin(current_url, location)
                    redirect_count += 1
                    continue

                response.raise_for_status()

                content_length = response.headers.get("content-length")
                if (
                    content_length
                    and int(content_length) > MAX_FILE_SIZE_BYTES
                ):
                    raise ValueError("文件大小超过 50 MB 限制。")

                temp_path: Path | None = None
                committed = False

                try:
                    descriptor, temp_name = tempfile.mkstemp(
                        prefix=".download_",
                        suffix=".tmp",
                        dir=target.parent,
                    )
                    temp_path = Path(temp_name)

                    size_bytes = 0
                    sha256 = hashlib.sha256()
                    prefix = b""

                    with os.fdopen(descriptor, "wb") as file:
                        async for chunk in response.aiter_bytes(
                            chunk_size=CHUNK_SIZE,
                        ):
                            size_bytes += len(chunk)

                            if size_bytes > MAX_FILE_SIZE_BYTES:
                                raise ValueError(
                                    "文件大小超过 50 MB 限制。"
                                )

                            if len(prefix) < 8:
                                prefix += chunk[: 8 - len(prefix)]

                            sha256.update(chunk)
                            file.write(chunk)

                    if expected_type == "pdf" and not _is_pdf(
                        content_type=response.headers.get("content-type"),
                        prefix=prefix,
                    ):
                        raise ValueError("下载内容不是有效的 PDF 文件。")

                    # 仅完整下载且校验成功时，才生成最终文件。
                    temp_path.replace(target)
                    committed = True

                    return {
                        "final_url": str(response.url),
                        "redirect_count": redirect_count,
                        "content_type": response.headers.get(
                            "content-type"
                        ),
                        "size_bytes": size_bytes,
                        "sha256": sha256.hexdigest(),
                    }

                finally:
                    # 无论超时、网络断开、文件过大、校验失败或取消任务，
                    # 均清理本次尝试产生的临时文件。
                    if (
                        not committed
                        and temp_path is not None
                        and temp_path.exists()
                    ):
                        temp_path.unlink()


async def _download_with_retry(**kwargs: Any) -> dict[str, Any]:
    for attempt in range(MAX_DOWNLOAD_ATTEMPTS):
        try:
            return await _download(**kwargs)

        except httpx.HTTPStatusError as exc:
            retryable = (
                exc.response.status_code in RETRYABLE_STATUS_CODES
            )
            if not retryable or attempt == MAX_DOWNLOAD_ATTEMPTS - 1:
                raise

        except httpx.RequestError:
            if attempt == MAX_DOWNLOAD_ATTEMPTS - 1:
                raise

        # 第 1、2 次失败后，分别等待 1 秒和 2 秒。
        await asyncio.sleep(2**attempt)

    raise RuntimeError("下载失败，已超过最大重试次数。")


async def download_file_handler(
    params: Dict[str, Any],
    context: ToolExecutionContext,
) -> Dict[str, Any]:
    del context

    args = DownloadFileArgs.model_validate(params)
    started_at = time.perf_counter()
    artifact_store = LocalArtifactStore()
    file_name = _ensure_extension(args)

    try:
        target = _target_path(
            artifact_store=artifact_store,
            save_dir=args.save_dir,
            file_name=file_name,
        )

        if target.exists() and not args.overwrite:
            existing = await asyncio.to_thread(
                _inspect_existing_file,
                target=target,
                expected_type=args.expected_type,
            )
            return {
                "ok": True,
                "url": args.url,
                "save_dir": args.save_dir,
                "file_name": file_name,
                "data": {
                    "saved_path": str(target),
                    "artifact_uri": artifact_store.to_artifact_uri(target),
                    "final_url": args.url,
                    "redirect_count": 0,
                    "content_type": None,
                    **existing,
                },
                "error": None,
                "metadata": {
                    "duration_ms": elapsed_ms(started_at),
                    "reused": True,
                },
            }

        downloaded = await _download_with_retry(
            url=args.url,
            target=target,
            expected_type=args.expected_type,
            overwrite=args.overwrite,
        )

        return {
            "ok": True,
            "url": args.url,
            "save_dir": args.save_dir,
            "file_name": file_name,
            "data": {
                "saved_path": str(target),
                "artifact_uri": artifact_store.to_artifact_uri(target),
                **downloaded,
            },
            "error": None,
            "metadata": {
                "duration_ms": elapsed_ms(started_at),
            },
        }

    except FileExistsError as exc:
        return _failure(
            args=args,
            started_at=started_at,
            code="FILE_ALREADY_EXISTS",
            message=str(exc),
        )

    except httpx.HTTPStatusError as exc:
        return _failure(
            args=args,
            started_at=started_at,
            code="UPSTREAM_HTTP_ERROR",
            message=f"下载站点返回 HTTP {exc.response.status_code}。",
            retryable=(
                exc.response.status_code in RETRYABLE_STATUS_CODES
            ),
            status_code=exc.response.status_code,
        )

    except httpx.RequestError:
        return _failure(
            args=args,
            started_at=started_at,
            code="DOWNLOAD_REQUEST_ERROR",
            message="下载失败，已超过最大重试次数。",
            retryable=True,
        )

    except ValueError as exc:
        return _failure(
            args=args,
            started_at=started_at,
            code="INVALID_DOWNLOAD_REQUEST",
            message=str(exc),
        )

    except OSError as exc:
        return _failure(
            args=args,
            started_at=started_at,
            code="STORAGE_ERROR",
            message=f"文件保存失败：{exc}",
        )


DOWNLOAD_FILE_TOOL = Tool(
    name="download_file",
    description=(
        "根据 HTTP 或 HTTPS 下载链接下载文件。"
        "必须提供保存目录 save_dir 和文件名 file_name。"
        "默认仅下载并校验 PDF，单个文件最大 50 MB。"
    ),
    input_schema=DownloadFileArgs.model_json_schema(),
    fn=download_file_handler,
    requires_confirmation=True,
)
