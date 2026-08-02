from __future__ import annotations

import asyncio
import time
from io import BytesIO
from typing import Any, Literal
from zipfile import BadZipFile, ZipFile
import httpx
from app.config import settings

MinerUModelVersion = Literal[
    "pipeline",
    "vlm",
    "MinerU-HTML",
]


class MinerUError(RuntimeError):
    pass


class MinerUTaskFailedError(MinerUError):
    pass


class MinerUTaskTimeoutError(MinerUError):
    pass


class MinerUResultError(MinerUError):
    pass


class MinerUClient:
    SINGLE_TASK_PATH = settings.MINERU_SINGLE_TASK_PATH
    BATCH_TASK_PATH = settings.MINERU_BATCH_TASK_PATH
    BATCH_RESULT_PATH = settings.MINERU_BATCH_RESULT_PATH
    BASE_URL = settings.MINERU_BASE_URL

    PENDING_STATES = {
        "waiting-file",
        "pending",
        "running",
        "converting",
    }

    def __init__(
        self,
        *,
        token: str,
        base_url: str = BASE_URL,
        request_timeout: float = 60,
        parse_timeout: float = 900,
        poll_interval: float = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.parse_timeout = parse_timeout
        self.poll_interval = poll_interval

        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout),
            follow_redirects=True,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
        )

    async def close(self) -> None:
        await self.http.aclose()

    async def submit_single_url(
        self,
        *,
        uri: str,
        data_id: str,
        model_version: MinerUModelVersion = "vlm",
        options: dict[str, Any] | None = None,
    ) -> str:
        body = {
            "url": uri,
            "data_id": data_id,
            "model_version": model_version,
            **(options or {}),
        }

        payload = await self._request_json(
            "POST",
            self.SINGLE_TASK_PATH,
            json=body,
        )

        task_id = payload.get("data", {}).get("task_id")

        if not task_id:
            raise MinerUResultError("MinerU response does not contain task_id")

        return str(task_id)

    async def wait_single(
        self,
        task_id: str,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.parse_timeout

        while time.monotonic() < deadline:
            payload = await self._request_json(
                "GET",
                f"{self.SINGLE_TASK_PATH}/{task_id}",
            )

            result = payload.get("data", {})
            state = result.get("state")

            if state == "done":
                return result

            if state == "failed":
                raise MinerUTaskFailedError(result.get("err_msg") or f"MinerU task failed: {task_id}")

            if state not in self.PENDING_STATES:
                raise MinerUResultError(f"Unexpected MinerU task state: {state}")

            await asyncio.sleep(self.poll_interval)

        raise MinerUTaskTimeoutError(f"MinerU task timed out: {task_id}")

    async def submit_batch_urls(
        self,
        *,
        files: list[dict[str, Any]],
        model_version: MinerUModelVersion = "vlm",
        options: dict[str, Any] | None = None,
    ) -> str:
        if not files:
            raise ValueError("files cannot be empty")

        if len(files) > 50:
            raise ValueError("MinerU batch supports at most 50 files")

        body = {
            "files": files,
            "model_version": model_version,
            **(options or {}),
        }

        payload = await self._request_json(
            "POST",
            self.BATCH_TASK_PATH,
            json=body,
        )

        batch_id = payload.get("data", {}).get("batch_id")

        if not batch_id:
            raise MinerUResultError("MinerU response does not contain batch_id")

        return str(batch_id)

    async def wait_batch(
        self,
        batch_id: str,
    ) -> list[dict[str, Any]]:
        deadline = time.monotonic() + self.parse_timeout

        while time.monotonic() < deadline:
            payload = await self._request_json(
                "GET",
                f"{self.BATCH_RESULT_PATH}/{batch_id}",
            )

            results = payload.get("data", {}).get("extract_result", [])

            if not results:
                await asyncio.sleep(self.poll_interval)
                continue

            if all(
                item.get("state") in {"done", "failed"}
                for item in results
            ):
                return results

            await asyncio.sleep(self.poll_interval)

        raise MinerUTaskTimeoutError(f"MinerU batch timed out: {batch_id}")

    async def download_result_zip(
        self,
        zip_url: str,
    ) -> bytes:
        response = await self.http.get(zip_url)
        response.raise_for_status()

        if not response.content:
            raise MinerUResultError("MinerU returned an empty ZIP")

        return response.content

    @staticmethod
    def extract_result_files(
        zip_content: bytes,
    ) -> dict[str, bytes]:
        try:
            with ZipFile(BytesIO(zip_content)) as archive:
                return {
                    name: archive.read(name)
                    for name in archive.namelist()
                    if not name.endswith("/")
                }
        except BadZipFile as error:
            raise MinerUResultError("MinerU result is not a valid ZIP") from error

    @staticmethod
    def extract_markdown(
        files: dict[str, bytes],
    ) -> str:
        for name, content in files.items():
            normalized = name.replace("\\", "/")
            basename = normalized.rsplit("/", 1)[-1]

            if basename == "full.md":
                markdown = content.decode("utf-8", errors="replace")
                if not markdown.strip():
                    raise MinerUResultError("MinerU returned empty Markdown")

                return markdown

        markdown_files = [
            content
            for name, content in files.items()
            if name.lower().endswith(".md")
        ]

        if len(markdown_files) == 1:
            return markdown_files[0].decode(
                "utf-8",
                errors="replace",
            )

        raise MinerUResultError("MinerU result does not contain full.md")

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await self.http.request(
            method,
            f"{self.base_url}/{path.lstrip('/')}",
            json=json,
        )
        response.raise_for_status()

        payload = response.json()

        if payload.get("code") != 0:
            raise MinerUError(
                "MinerU API error: "
                f"code={payload.get('code')}, "
                f"message={payload.get('msg')}"
            )

        return payload