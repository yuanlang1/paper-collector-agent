from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.store.postgres.base import PostgresIndexConfig
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.rag.embeddings.siliconflow import SiliconFlowEmbeddings

DENSE_EMBEDDING_DIMS = 1024


class MemorySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        extra="ignore",
    )

    DATABASE_POSTGRES_URL: SecretStr
    SILICONFLOW_API_KEY: SecretStr
    SILICONFLOW_EMBEDDING_BASE_URL: str = "https://api.siliconflow.cn/v1"
    DENSE_EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"


def memory_index(settings: MemorySettings) -> PostgresIndexConfig:
    return {
        "dims": DENSE_EMBEDDING_DIMS,
        "embed": SiliconFlowEmbeddings(
            api_key=settings.SILICONFLOW_API_KEY.get_secret_value(),
            model=settings.DENSE_EMBEDDING_MODEL_NAME,
            base_url=settings.SILICONFLOW_EMBEDDING_BASE_URL,
        ),
        "fields": ["content"],
        "distance_type": "cosine",
        "ann_index_config": {
            "kind": "hnsw",
            "vector_type": "halfvec",
            "m": 16,
            "ef_construction": 64,
        },
    }


@asynccontextmanager
async def open_memory_store(
    settings: MemorySettings | None = None,
) -> AsyncIterator[AsyncPostgresStore]:
    settings = settings or MemorySettings()

    async with AsyncPostgresStore.from_conn_string(
        settings.DATABASE_POSTGRES_URL.get_secret_value(),
        index=memory_index(settings),
        ttl={"omit_expired": True},
        pool_config={
            "min_size": 1,
            "max_size": 5,
            "kwargs": {
                "options": (
                    "-c search_path=public "
                    "-c hnsw.ef_search=100 "
                    "-c hnsw.iterative_scan=strict_order"
                )
            },
        },
    ) as store:
        yield store
