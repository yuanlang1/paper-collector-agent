import argparse
import asyncio
import json
import sys
from typing import cast

from langchain_core.embeddings import Embeddings
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from app.memory.store import (
    DENSE_EMBEDDING_DIMS,
    MemorySettings,
    memory_index,
)


async def initialize(*, check_embedding: bool = False) -> None:
    settings = MemorySettings()
    index = memory_index(settings)

    if check_embedding:
        embeddings = cast(Embeddings, index["embed"])
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

        store = AsyncPostgresStore(conn, index=index)
        await store.setup()

        cursor = await conn.execute("""
            SELECT format_type(
                       a.atttypid,
                       a.atttypmod
                   ) AS column_type,
                   i.indisvalid AS valid,
                   pg_get_indexdef(i.indexrelid) AS definition
            FROM pg_attribute a
            JOIN pg_index i ON i.indrelid = a.attrelid
            WHERE a.attrelid = to_regclass('public.store_vectors')
              AND a.attname = 'embedding'
              AND NOT a.attisdropped
              AND i.indexrelid = to_regclass(
                  'public.store_vectors_embedding_idx'
              )
        """)

        row = await cursor.fetchone()

        if row is None or not row["valid"]:
            raise RuntimeError("Memory vector index is missing or invalid")

        if row["column_type"] != "halfvec(1024)":
            raise RuntimeError(f"Expected halfvec(1024), got {row['column_type']}")

        if (
            "USING hnsw" not in row["definition"]
            or "halfvec_cosine_ops" not in row["definition"]
        ):
            raise RuntimeError("Expected an HNSW halfvec_cosine_ops index")

        print(
            json.dumps(
                {
                    "status": "ok",
                    "model": settings.DENSE_EMBEDDING_MODEL_NAME,
                    "dimensions": DENSE_EMBEDDING_DIMS,
                    "embedding_checked": check_embedding,
                    "column_type": row["column_type"],
                    "index": row["definition"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-embedding", action="store_true")
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(
        initialize(
            check_embedding=args.check_embedding,
        )
    )
