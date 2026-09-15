from __future__ import annotations

import asyncio
from datetime import timedelta
import re
from pathlib import Path
from typing import Protocol

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
    ) -> str | None: ...

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


class AliyunOssObjectStore:
    """Stores RAG source files and produces temporary PDF download URLs."""

    def __init__(self) -> None:
        self._client = None
        self._oss = None

    def _get_client(self):
        if self._client is not None:
            return self._client, self._oss

        if (
            not settings.OSS_BUCKET
            or not settings.OSS_REGION
            or not settings.OSS_ACCESS_KEY_ID
            or not settings.OSS_ACCESS_KEY_SECRET
        ):
            raise RuntimeError(
                "OSS_BUCKET, OSS_REGION, OSS_ACCESS_KEY_ID, and "
                "OSS_ACCESS_KEY_SECRET must be configured"
            )

        try:
            import alibabacloud_oss_v2 as oss
        except ImportError as exc:
            raise RuntimeError("alibabacloud-oss-v2 is not installed") from exc

        config = oss.config.load_default()
        config.credentials_provider = (
            oss.credentials.StaticCredentialsProvider(
                settings.OSS_ACCESS_KEY_ID,
                settings.OSS_ACCESS_KEY_SECRET,
                settings.OSS_SESSION_TOKEN or None,
            )
        )
        config.region = settings.OSS_REGION
        if settings.OSS_ENDPOINT:
            config.endpoint = settings.OSS_ENDPOINT

        self._client = oss.Client(config)
        self._oss = oss
        return self._client, self._oss

    def _upload_pdf_sync(
        self,
        *,
        source_path: Path,
        object_name: str,
        sha256: str,
    ) -> None:
        if not source_path.is_file():
            raise ValueError("OSS upload source PDF does not exist")

        client, oss = self._get_client()
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
            self._upload_pdf_sync,
            source_path=source_path,
            object_name=object_name,
            sha256=sha256,
        )

    async def get_pdf_temporary_url(
        self,
        *,
        pdf_sha256: str,
    ) -> str | None:
        if not settings.OSS_ENABLED:
            return None

        object_name = build_pdf_object_name(
            prefix=settings.OSS_PREFIX,
            pdf_sha256=pdf_sha256,
        )
        if object_name is None:
            return None

        return await asyncio.to_thread(
            self._get_pdf_temporary_url_sync,
            object_name=object_name,
        )

    def _get_pdf_temporary_url_sync(
        self,
        *,
        object_name: str,
    ) -> str:
        client, oss = self._get_client()
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
            raise RuntimeError("OSS returned an empty presigned PDF URL")
        return str(result.url)

    async def upload_markdown(
        self,
        *,
        markdown: str,
        pdf_sha256: str,
    ) -> None:
        if not settings.OSS_ENABLED or not markdown:
            return

        sha256 = normalize_pdf_sha256(pdf_sha256)
        if sha256 is None:
            return

        object_name = build_markdown_object_name(
            prefix=settings.OSS_PREFIX,
            pdf_sha256=sha256,
        )
        if object_name is None:
            return

        await asyncio.to_thread(
            self._upload_markdown_sync,
            markdown=markdown,
            object_name=object_name,
            sha256=sha256,
        )

    def _upload_markdown_sync(
        self,
        *,
        markdown: str,
        object_name: str,
        sha256: str,
    ) -> None:
        client, oss = self._get_client()
        client.put_object(
            oss.PutObjectRequest(
                bucket=settings.OSS_BUCKET,
                key=object_name,
                body=markdown.encode("utf-8"),
                content_type="text/markdown; charset=utf-8",
                metadata={"sha256": sha256},
            )
        )
