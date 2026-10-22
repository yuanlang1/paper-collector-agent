"""Initialize PostgreSQL memory storage without starting the application."""

import argparse
import asyncio
import json
import sys

from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.rag.embeddings.siliconflow import SiliconFlowEmbeddings

DENSE_EMBEDDING_DIMS = 1024


class MemorySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_POSTGRES_URL: SecretStr
    SILICONFLOW_API_KEY: SecretStr
    SILICONFLOW_EMBEDDING_BASE_URL: str = "https://api.siliconflow.cn/v1"
    DENSE_EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"


async def initialize(*, check_embedding: bool = False) -> None:
    settings = MemorySettings()
    embeddings = SiliconFlowEmbeddings(
        api_key=settings.SILICONFLOW_API_KEY.get_secret_value(),
        model=settings.DENSE_EMBEDDING_MODEL_NAME,
        base_url=settings.SILICONFLOW_EMBEDDING_BASE_URL,
    )
    if check_embedding:
        vector = await embeddings.aembed_query("记忆向量维度检查")
        if len(vector) != DENSE_EMBEDDING_DIMS:
            raise ValueError(
                f"Expected {DENSE_EMBEDDING_DIMS} dimensions, got {len(vector)}"
            )

    async with await AsyncConnection.connect(
        settings.DATABASE_POSTGRES_URL.get_secret_value(),
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    ) as conn:
        await conn.execute("SET search_path TO public")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS zhparser WITH SCHEMA public")
        await conn.execute("""
            DO $body$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_ts_config AS config
                    JOIN pg_namespace AS ns ON ns.oid = config.cfgnamespace
                    WHERE config.cfgname = 'memory_zh'
                        AND ns.nspname = 'public'
                ) THEN
                    CREATE TEXT SEARCH CONFIGURATION public.memory_zh
                        (PARSER = public.zhparser);
                    ALTER TEXT SEARCH CONFIGURATION public.memory_zh
                        ADD MAPPING FOR n, v, a, i, e, l WITH pg_catalog.simple;
                END IF;
            END;
            $body$;
        """)

        cur = await conn.execute("""
            SELECT format_type(atttypid, atttypmod) AS vector_type
            FROM pg_attribute
            WHERE attrelid = to_regclass('public.store_vectors')
                AND attname = 'embedding' AND NOT attisdropped
        """)
        column = await cur.fetchone()
        if column and column["vector_type"] != "vector(1024)":
            raise ValueError("Existing vector column needs a separate dimension migration")

        store = AsyncPostgresStore(
            conn,
            index={
                "dims": DENSE_EMBEDDING_DIMS,
                "embed": embeddings,
                "fields": ["content"],
                "distance_type": "cosine",
                "ann_index_config": {
                    "kind": "hnsw",
                    "vector_type": "vector",
                    "m": 16,
                    "ef_construction": 64,
                },
            },
        )
        await store.setup()
        await conn.execute("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS memory_store_fts_idx
            ON public.store USING gin (
                jsonb_to_tsvector(
                    'public.memory_zh'::regconfig,
                    value -> 'content',
                    '["string"]'::jsonb
                )
            )
        """)

        cur = await conn.execute("""
            SELECT index_class.relname AS name, index_info.indisvalid AS valid,
                pg_get_indexdef(index_class.oid) AS definition
            FROM pg_index AS index_info
            JOIN pg_class AS index_class
                ON index_class.oid = index_info.indexrelid
            JOIN pg_namespace AS ns ON ns.oid = index_class.relnamespace
            WHERE ns.nspname = 'public' AND index_class.relname IN (
                'store_vectors_embedding_idx', 'memory_store_fts_idx'
            )
        """)
        indexes = await cur.fetchall()
        if len(indexes) != 2 or not all(item["valid"] for item in indexes):
            raise RuntimeError("Memory indexes are missing or invalid")
        vector_index = next(
            item for item in indexes if item["name"] == "store_vectors_embedding_idx"
        )
        if "USING hnsw" not in vector_index["definition"] or (
            "vector_cosine_ops" not in vector_index["definition"]
        ):
            raise RuntimeError("Existing vector index needs a separate HNSW migration")

        print(json.dumps({
            "status": "ok",
            "model": settings.DENSE_EMBEDDING_MODEL_NAME,
            "dimensions": DENSE_EMBEDDING_DIMS,
            "embedding_checked": check_embedding,
            "indexes": indexes,
        }, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-embedding", action="store_true")
    args = parser.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(initialize(check_embedding=args.check_embedding))
