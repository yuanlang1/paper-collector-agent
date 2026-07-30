from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from app.config import settings


class LightRAGConfigurationError(RuntimeError):
    """Raised when the LightRAG connection has not been configured."""


class LightRAGProtocolError(RuntimeError):
    """Raised when LightRAG returns an unexpected response payload."""


class LightRAGClient:
    """Small HTTP adapter for LightRAG document ingestion APIs."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_base_url = (base_url or settings.LIGHTRAG_BASE_URL).rstrip("/")
        self.base_url = resolved_base_url
        self.api_key = api_key if api_key is not None else settings.LIGHTRAG_API_KEY
        self._client = client
        self._timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.LIGHTRAG_REQUEST_TIMEOUT_SECONDS
        )

    def _headers(self) -> dict[str, str]:
        if not self.base_url:
            raise LightRAGConfigurationError("LIGHTRAG_BASE_URL is not configured")
        if not self.api_key:
            raise LightRAGConfigurationError("LIGHTRAG_API_KEY is not configured")
        return {"X-API-Key": self.api_key}

    async def upload_file(self, file_path: str | Path) -> str:
        """Upload one document and return LightRAG's asynchronous track ID."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"LightRAG upload file does not exist: {path}")

        with path.open("rb") as file:
            response = await self._request(
                "POST",
                "/documents/upload",
                files={"file": (path.name, file, "application/pdf")},
            )
        return self._extract_track_id(response)

    async def get_track_status(self, track_id: str) -> dict[str, Any]:
        """Return the current processing state for an upload track ID."""
        if not track_id.strip():
            raise ValueError("track_id must not be blank")
        payload = await self._request("GET", f"/documents/track_status/{track_id}")
        return self._unwrap_data(payload)

    async def health(self) -> dict[str, Any]:
        """Return LightRAG health information for operational checks."""
        payload = await self._request("GET", "/health")
        return self._unwrap_data(payload)

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        request_kwargs = {"headers": self._headers(), **kwargs}
        if self._client is not None:
            response = await self._client.request(method, f"{self.base_url}{path}", **request_kwargs)
        else:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.request(method, f"{self.base_url}{path}", **request_kwargs)

        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise LightRAGProtocolError("LightRAG response must be a JSON object")
        return payload

    @staticmethod
    def _unwrap_data(payload: dict[str, Any]) -> dict[str, Any]:
        data = payload.get("data", payload)
        if not isinstance(data, dict):
            raise LightRAGProtocolError("LightRAG response data must be a JSON object")
        return data

    @classmethod
    def _extract_track_id(cls, payload: dict[str, Any]) -> str:
        data = cls._unwrap_data(payload)
        track_id = data.get("track_id") or payload.get("track_id")
        if not isinstance(track_id, str) or not track_id.strip():
            raise LightRAGProtocolError("LightRAG upload response does not contain track_id")
        return track_id
