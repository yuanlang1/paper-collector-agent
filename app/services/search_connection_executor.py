from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from sqlalchemy import select

from app.core.identifiers import normalize_user_id
from app.db.models.search_connection import SearchConnection
from app.infrastructure.http import http_manager
from app.llm.tools.search_tools.arxiv.search_arxiv import ArxivSearchArgs
from app.llm.tools.search_tools.dblp.search_dblp import DblpSearchArgs
from app.llm.tools.search_tools.google_scholar.search_google_scholar import (
    GoogleScholarSearchArgs,
)
from app.utils.search_adapters.contracts import (
    DEFAULT_INITIAL_LIMITS,
    SearchSettings,
    source_arguments,
)
from app.utils.search_adapters.registry import get_adapter, supported_provider_ids


class SearchConnectionError(ValueError):
    pass


_QUERY_SCHEMAS = {
    "arxiv": (ArxivSearchArgs, "max_results", "start"),
    "dblp": (DblpSearchArgs, "h", "f"),
    "google_scholar": (GoogleScholarSearchArgs, "num", "start"),
}


def validate_query(provider_id: str, query_args: dict) -> None:
    schema, page_field, offset_field = _QUERY_SCHEMAS[provider_id]
    arguments = source_arguments(
        query_args,
        None,
        1,
        1,
        page_field=page_field,
        offset_field=offset_field,
    )
    schema.model_validate(arguments)


@dataclass(frozen=True)
class _ConnectionConfig:
    provider_id: str
    settings: dict[str, Any]
    enabled: bool
    status: str
    version: str
    api_key: str | None = field(repr=False)


class SearchConnectionExecutor:
    def __init__(self, session_factory=None, http_client=None):
        self.session_factory = session_factory
        self.http_client = http_client

    def configure(self, session_factory) -> None:
        self.session_factory = session_factory

    def reset(self) -> None:
        self.session_factory = None

    async def _load(self, user_id, providers):
        if self.session_factory is None:
            raise SearchConnectionError("检索连接执行服务未初始化")
        user_id = normalize_user_id(user_id)[1]
        async with self.session_factory() as db, db.begin():
            rows = await db.scalars(
                select(SearchConnection).where(
                    SearchConnection.user_id == user_id,
                    SearchConnection.provider_id.in_(providers),
                )
            )
            return {
                row.provider_id: _ConnectionConfig(
                    provider_id=row.provider_id,
                    settings=dict(row.settings),
                    enabled=row.enabled,
                    status=row.status,
                    api_key=row.api_key,
                    version=sha256(
                        f"{row.id}:{row.modification_time.isoformat()}".encode()
                    ).hexdigest(),
                )
                for row in rows
            }

    @staticmethod
    def _check(config):
        if not config.enabled or config.status == "invalid":
            raise SearchConnectionError(
                f"{config.provider_id}: connection is disabled or invalid"
            )
        if config.provider_id == "google_scholar" and not config.api_key:
            raise SearchConnectionError("google_scholar: api_key is required")
        try:
            return SearchSettings.model_validate({
                "initial_limit": DEFAULT_INITIAL_LIMITS[config.provider_id],
                **config.settings,
            })
        except ValueError:
            raise SearchConnectionError(
                f"{config.provider_id}: invalid connection settings"
            ) from None

    async def get_available_providers(self, *, user_id, provider_ids=None):
        providers = (
            provider_ids if provider_ids is not None else supported_provider_ids()
        )
        for provider in providers:
            get_adapter(provider)
        configs = await self._load(user_id, providers)
        available, warnings = {}, []
        for provider in providers:
            try:
                config = configs.get(provider)
                if config is None:
                    raise SearchConnectionError(
                        f"{provider}: connection is not configured"
                    )
                options = self._check(config)
            except SearchConnectionError as exc:
                if provider_ids is not None:
                    raise
                if config is not None:
                    warnings.append(str(exc))
                continue
            available[provider] = {
                "initial_limit": options.initial_limit,
                "configuration_version": config.version,
            }
        if not available:
            raise SearchConnectionError("没有可用检索连接，请先配置并启用检索来源")
        return available, warnings

    async def search(self, *, user_id, task):
        provider = task["provider_id"]
        adapter = get_adapter(provider)
        config = (await self._load(user_id, [provider])).get(provider)
        if config is None:
            raise SearchConnectionError(f"{provider}: connection is not configured")
        self._check(config)
        if config.version != task["configuration_version"]:
            raise SearchConnectionError(
                f"{provider}: connection changed; restart search"
            )
        return await adapter.search(
            query_args=task["query_args"],
            credentials={"api_key": config.api_key}
            if provider == "google_scholar"
            else {},
            cursor=task["cursor"],
            limit=task["limit"],
            max_pages=task["max_pages"],
            http_client=self.http_client
            if self.http_client is not None
            else http_manager.search_client,
        )


search_executor = SearchConnectionExecutor()
