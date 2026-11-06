from __future__ import annotations

import asyncio
import hashlib
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from threading import Lock
from typing import Protocol

from app.common.exceptions import (
    BadRequestException,
    NotFoundException,
    ServiceUnavailableException,
)
from app.config import settings

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PRESIGNED_URL_EXPIRES = timedelta(days=1)


def normalize_pdf_sha256(value: object) -> str | None:
    checksum = str(value or "").strip().lower()
    if not _SHA256.fullmatch(checksum):
        return None
    return checksum


def _build_object_name(
    *,
    prefix: str,
    directory: str,
    pdf_sha256: object,
    extension: str,
) -> str | None:
    checksum = normalize_pdf_sha256(pdf_sha256)
    if checksum is None:
        return None

    normalized_prefix = prefix.strip().strip("/")
    return "/".join(
        part
        for part in (
            normalized_prefix,
            directory,
            f"{checksum}.{extension}",
        )
        if part
    )


def build_pdf_object_name(*, prefix: str, pdf_sha256: object) -> str | None:
    return _build_object_name(
        prefix=prefix,
        directory="pdf",
        pdf_sha256=pdf_sha256,
        extension="pdf",
    )


def build_markdown_object_name(
    *,
    prefix: str,
    pdf_sha256: object,
) -> str | None:
    return _build_object_name(
        prefix=prefix,
        directory="md",
        pdf_sha256=pdf_sha256,
        extension="md",
    )


class OssObjectStore(Protocol):
    async def get_pdf_temporary_url(
        self,
        *,
        pdf_sha256: str,
    ) -> str: ...

    async def upload_pdf(
        self,
        *,
        source_path: Path,
        object_name: str,
        sha256: str,
    ) -> None: ...

    async def upload_markdown(
        self,
        *,
        markdown: str,
        pdf_sha256: str,
    ) -> None: ...

    async def get_markdown(
        self,
        *,
        pdf_sha256: str,
    ) -> MarkdownObject: ...


@dataclass(frozen=True, slots=True)
class MarkdownObject:
    content: str
    object_name: str
    content_sha256: str


class AliyunOssObjectStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._transport = None
        self._client = None
        self._oss = None

    def _run_sync(self, operation, **kwargs):
        with self._lock:
            return operation(**kwargs)

    async def close(self) -> None:
        await asyncio.to_thread(self._run_sync, self._close_sync)

    def _close_sync(self) -> None:
        if self._transport is not None:
            self._transport.close()
        self._client = self._oss = self._transport = None

    def _get_client(self):
        if not settings.OSS_ENABLED:
            raise ServiceUnavailableException("OSS 服务未启用。")

        if self._client is not None:
            return self._client, self._oss

        if (
            not settings.OSS_BUCKET
            or not settings.OSS_REGION
            or not settings.OSS_ACCESS_KEY_ID
            or not settings.OSS_ACCESS_KEY_SECRET
        ):
            raise ServiceUnavailableException("OSS 服务配置不完整。")

        try:
            import alibabacloud_oss_v2 as oss
        except ImportError as exc:
            raise ServiceUnavailableException("OSS SDK 未安装。") from exc

        config = oss.config.load_default()
        config.credentials_provider = oss.credentials.StaticCredentialsProvider(
            settings.OSS_ACCESS_KEY_ID,
            settings.OSS_ACCESS_KEY_SECRET,
            settings.OSS_SESSION_TOKEN or None,
        )
        config.region = settings.OSS_REGION
        if settings.OSS_ENDPOINT:
            config.endpoint = settings.OSS_ENDPOINT

        self._transport = oss.transport.RequestsHttpClient()
        config.http_client = self._transport
        self._client = oss.Client(config)
        self._oss = oss
        return self._client, self._oss

    @staticmethod
    @contextmanager
    def _handle_oss_errors():
        from alibabacloud_oss_v2.exceptions import (
            OperationError,
            RequestError,
            ResponseError,
            ServiceError,
        )

        try:
            yield
        except (OperationError, ServiceError, RequestError, ResponseError) as exc:
            error = exc.unwrap() if isinstance(exc, OperationError) else exc
            if (
                isinstance(error, ServiceError)
                and error.status_code == 404
                and error.code == "NoSuchKey"
            ):
                raise NotFoundException("OSS 对象不存在。") from exc
            if isinstance(error, (ServiceError, RequestError, ResponseError)):
                raise ServiceUnavailableException("OSS 服务暂不可用。") from exc
            raise

    def _upload_pdf_sync(
        self,
        *,
        source_path: Path,
        object_name: str,
        sha256: str,
    ) -> None:
        if not source_path.is_file():
            raise NotFoundException("待上传的 PDF 文件不存在。")

        client, oss = self._get_client()
        with self._handle_oss_errors():
            client.put_object_from_file(
                oss.PutObjectRequest(
                    bucket=settings.OSS_BUCKET,
                    key=object_name,
                    content_type="application/pdf",
                    metadata={"sha256": sha256},
                ),
                str(source_path),
            )

    async def upload_pdf(
        self,
        *,
        source_path: Path,
        object_name: str,
        sha256: str,
    ) -> None:
        await asyncio.to_thread(
            self._run_sync,
            self._upload_pdf_sync,
            source_path=source_path,
            object_name=object_name,
            sha256=sha256,
        )

    async def get_pdf_temporary_url(
        self,
        *,
        pdf_sha256: str,
    ) -> str:
        object_name = build_pdf_object_name(
            prefix=settings.OSS_PREFIX,
            pdf_sha256=pdf_sha256,
        )
        if object_name is None:
            raise BadRequestException("pdf_sha256 必须是 64 位十六进制字符串。")

        return await asyncio.to_thread(
            self._run_sync,
            self._get_pdf_temporary_url_sync,
            object_name=object_name,
        )

    def _get_pdf_temporary_url_sync(
        self,
        *,
        object_name: str,
    ) -> str:
        client, oss = self._get_client()
        with self._handle_oss_errors():
            client.head_object(
                oss.HeadObjectRequest(
                    bucket=settings.OSS_BUCKET,
                    key=object_name,
                )
            )
            result = client.presign(
                oss.GetObjectRequest(
                    bucket=settings.OSS_BUCKET,
                    key=object_name,
                ),
                expires=_PRESIGNED_URL_EXPIRES,
            )
        if not result.url:
            raise ServiceUnavailableException("OSS 未返回有效的 PDF 临时链接。")
        return str(result.url)

    async def upload_markdown(
        self,
        *,
        markdown: str,
        pdf_sha256: str,
    ) -> None:
        if not markdown:
            return

        sha256 = normalize_pdf_sha256(pdf_sha256)
        if sha256 is None:
            raise BadRequestException("pdf_sha256 必须是 64 位十六进制字符串。")

        object_name = build_markdown_object_name(
            prefix=settings.OSS_PREFIX,
            pdf_sha256=sha256,
        )
        if object_name is None:
            raise BadRequestException("pdf_sha256 必须是 64 位十六进制字符串。")

        await asyncio.to_thread(
            self._run_sync,
            self._upload_markdown_sync,
            markdown=markdown,
            object_name=object_name,
            sha256=sha256,
        )

    async def get_markdown(
        self,
        *,
        pdf_sha256: str,
    ) -> MarkdownObject:
        object_name = build_markdown_object_name(
            prefix=settings.OSS_PREFIX,
            pdf_sha256=pdf_sha256,
        )
        if object_name is None:
            raise BadRequestException("pdf_sha256 必须是 64 位十六进制字符串。")

        return await asyncio.to_thread(
            self._run_sync,
            self._get_markdown_sync,
            object_name=object_name,
        )

    def _get_markdown_sync(
        self,
        *,
        object_name: str,
    ) -> MarkdownObject:
        client, oss = self._get_client()
        with self._handle_oss_errors():
            result = client.get_object(
                oss.GetObjectRequest(
                    bucket=settings.OSS_BUCKET,
                    key=object_name,
                )
            )
            try:
                content = result.body.read().decode("utf-8")
            finally:
                result.body.close()

        if not content.strip():
            raise NotFoundException("论文没有可读取的 OSS Markdown。")

        return MarkdownObject(
            content=content,
            object_name=object_name,
            content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )

    def _upload_markdown_sync(
        self,
        *,
        markdown: str,
        object_name: str,
        sha256: str,
    ) -> None:
        client, oss = self._get_client()
        with self._handle_oss_errors():
            client.put_object(
                oss.PutObjectRequest(
                    bucket=settings.OSS_BUCKET,
                    key=object_name,
                    body=markdown.encode("utf-8"),
                    content_type="text/markdown; charset=utf-8",
                    metadata={"sha256": sha256},
                )
            )


oss_store = AliyunOssObjectStore()
