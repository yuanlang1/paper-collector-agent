from contextlib import AsyncExitStack

import httpx

from app.config import settings


class HttpManager:
    def __init__(self) -> None:
        self._mineru_client = None
        self._resources: AsyncExitStack | None = None
        self._search_client: httpx.AsyncClient | None = None
        self._paper_service_client: httpx.AsyncClient | None = None
        self._crossref_client: httpx.AsyncClient | None = None

    @property
    def search_client(self) -> httpx.AsyncClient:
        if self._search_client is None:
            if self._resources is None:
                raise RuntimeError("HTTP manager is not initialized")
            self._search_client = httpx.AsyncClient(timeout=30.0)
            self._resources.push_async_callback(self._search_client.aclose)
        return self._search_client

    @property
    def paper_service_client(self) -> httpx.AsyncClient:
        if self._paper_service_client is None:
            raise RuntimeError("HTTP manager is not initialized")
        return self._paper_service_client

    @property
    def crossref_client(self) -> httpx.AsyncClient:
        if self._crossref_client is None:
            if self._resources is None:
                raise RuntimeError("HTTP manager is not initialized")
            self._crossref_client = httpx.AsyncClient(
                timeout=settings.CROSSREF_TIMEOUT_SECONDS,
                headers={
                    "User-Agent": (
                        "paper-collector-agent/1.0 "
                        f"(mailto:{settings.CROSSREF_MAILTO or 'not-configured'})"
                    ),
                },
            )
            self._resources.push_async_callback(self._crossref_client.aclose)
        return self._crossref_client

    @property
    def mineru_client(self):
        if self._resources is None:
            raise RuntimeError("HTTP manager is not initialized")
        if self._mineru_client is None:
            from app.infrastructure.mineru.mineru_client import MinerUClient

            self._mineru_client = MinerUClient(token=settings.MINERU_TOKEN)
            self._resources.push_async_callback(self._mineru_client.close)
        return self._mineru_client

    async def start(self) -> None:
        if self._resources is not None:
            return
        async with AsyncExitStack() as resources:
            paper_service = await resources.enter_async_context(
                httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        settings.PAPER_SERVICE_TIMEOUT_SECONDS,
                        connect=settings.PAPER_SERVICE_CONNECT_TIMEOUT_SECONDS,
                    ),
                )
            )
            self._paper_service_client = paper_service
            self._resources = resources.pop_all()

    async def close(self) -> None:
        resources, self._resources = self._resources, None
        self._search_client = self._paper_service_client = None
        self._crossref_client = None
        self._mineru_client = None
        if resources is not None:
            await resources.aclose()


http_manager = HttpManager()
