from __future__ import annotations

import asyncio
import math
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings


class SiliconFlowEmbeddings(Embeddings):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.siliconflow.cn/v1",
        timeout_seconds: float = 30.0,
        batch_size: int = 32,
        max_retries: int = 3,
    ) -> None:
        if not api_key.strip():
            raise ValueError("SILICONFLOW_API_KEY is required")
        if not model.strip():
            raise ValueError("SILICONFLOW_EMBEDDING_MODEL is required")
        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")
        if max_retries < 1:
            raise ValueError("max_retries must be greater than 0")

        self.api_key = api_key
        self.model = model
        self.endpoint = f"{base_url.rstrip('/')}/embeddings"
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size
        self.max_retries = max_retries

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return self._embed_documents_sync(texts)

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        result: list[list[float]] = []

        for batch in self._chunked(texts):
            result.extend(await self._embed_batch_async(batch))

        return result

    async def aembed_query(
        self,
        text: str,
    ) -> list[float]:
        return (await self.aembed_documents([text]))[0]

    def _embed_documents_sync(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        result: list[list[float]] = []

        with httpx.Client(timeout=self.timeout_seconds) as client:
            for batch in self._chunked(texts):
                result.extend(self._embed_batch_sync(client, batch))

        return result

    async def _embed_batch_async(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(
                        self.endpoint,
                        headers=self._headers(),
                        json=self._payload(texts),
                    )

                    if response.status_code in {429, 503, 504}:
                        await self._sleep_before_retry(attempt, response)
                        continue

                    response.raise_for_status()
                    return self._parse_embeddings(response.json(), len(texts))

                except httpx.TransportError:
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(0.5 * (2 ** attempt))

        raise RuntimeError("SiliconFlow embedding request failed")

    def _embed_batch_sync(
        self,
        client: httpx.Client,
        texts: list[str],
    ) -> list[list[float]]:
        for attempt in range(self.max_retries):
            try:
                response = client.post(
                    self.endpoint,
                    headers=self._headers(),
                    json=self._payload(texts),
                )

                if response.status_code in {429, 503, 504}:
                    if attempt == self.max_retries - 1:
                        response.raise_for_status()
                    continue

                response.raise_for_status()
                return self._parse_embeddings(response.json(), len(texts))

            except httpx.TransportError:
                if attempt == self.max_retries - 1:
                    raise

        raise RuntimeError("SiliconFlow embedding request failed")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self,
        texts: list[str],
    ) -> dict[str, Any]:
        if any(not text.strip() for text in texts):
            raise ValueError("embedding input cannot contain empty text")

        return {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }

    @staticmethod
    def _parse_embeddings(
        payload: dict[str, Any],
        expected_count: int,
    ) -> list[list[float]]:
        data = payload.get("data")

        if not isinstance(data, list) or len(data) != expected_count:
            raise RuntimeError("SiliconFlow returned an invalid embedding response")

        ordered_data = sorted(data, key=lambda item: int(item["index"]))
        vectors = [
            [float(value) for value in item["embedding"]]
            for item in ordered_data
        ]

        return [
            SiliconFlowEmbeddings._normalize(vector)
            for vector in vectors
        ]

    @staticmethod
    def _normalize(
        vector: list[float],
    ) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))

        if norm == 0:
            raise RuntimeError("SiliconFlow returned a zero embedding vector")

        return [
            value / norm
            for value in vector
        ]

    def _chunked(
        self,
        texts: list[str],
    ) -> list[list[str]]:
        return [
            texts[index:index + self.batch_size]
            for index in range(0, len(texts), self.batch_size)
        ]

    async def _sleep_before_retry(
        self,
        attempt: int,
        response: httpx.Response,
    ) -> None:
        if attempt == self.max_retries - 1:
            response.raise_for_status()

        retry_after = response.headers.get("Retry-After")

        try:
            delay = float(retry_after) if retry_after else 0.5 * (2 ** attempt)
        except ValueError:
            delay = 0.5 * (2 ** attempt)

        await asyncio.sleep(delay)