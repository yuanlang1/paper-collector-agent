from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import AsyncQdrantClient, models

from app.config import settings
from app.rag.embeddings.siliconflow import SiliconFlowEmbeddings

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"


def user_filter(user_id: str) -> models.Filter:
    user_id = user_id.strip()
    if not user_id:
        raise ValueError("user_id cannot be empty")
    return models.Filter(
        must=[
            models.FieldCondition(
                key="user_id", match=models.MatchValue(value=user_id)
            ),
        ]
    )


@lru_cache(maxsize=1)
def get_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None,
    )


@lru_cache(maxsize=1)
def get_dense_embeddings() -> Embeddings:
    return SiliconFlowEmbeddings(
        api_key=settings.SILICONFLOW_API_KEY,
        model=settings.SILICONFLOW_EMBEDDING_MODEL,
        base_url=settings.SILICONFLOW_EMBEDDING_BASE_URL,
        timeout_seconds=settings.SILICONFLOW_EMBEDDING_TIMEOUT_SECONDS,
        batch_size=settings.SILICONFLOW_EMBEDDING_BATCH_SIZE,
        max_retries=settings.SILICONFLOW_EMBEDDING_MAX_RETRIES,
    )


async def close_index_resources() -> None:
    if get_qdrant_client.cache_info().currsize:
        await get_qdrant_client().close()
        get_qdrant_client.cache_clear()

    get_dense_embeddings.cache_clear()


@dataclass(frozen=True, slots=True)
class IndexConstructionResult:
    collection_name: str
    indexed_count: int
    embedding_dimension: int


class BaseQdrantIndexConstructionModule(ABC):
    def __init__(
        self,
        *,
        collection_name: str,
        batch_size: int = 32,
        distance: models.Distance = models.Distance.COSINE,
        client: AsyncQdrantClient | None = None,
        dense_embeddings: Embeddings | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")

        self.collection_name = collection_name
        self.batch_size = batch_size
        self.distance = distance

        self.client = client or get_qdrant_client()
        self.dense_embeddings = dense_embeddings or get_dense_embeddings()

    async def build_index(
        self,
        documents: list[Document],
    ) -> IndexConstructionResult:
        logger.info(f"Build index number: {len(documents)}")

        if not documents:
            return IndexConstructionResult(
                collection_name=self.collection_name,
                indexed_count=0,
                embedding_dimension=0,
            )

        indexed_count = 0
        embedding_dimension = 0

        for batch in self._chunked(documents, self.batch_size):
            texts = [self._build_embedding_text(document) for document in batch]

            dense_vectors = await self.dense_embeddings.aembed_documents(texts)

            if embedding_dimension == 0:
                embedding_dimension = len(dense_vectors[0])

                await self._ensure_collection(embedding_dimension)

            points = [
                models.PointStruct(
                    id=self._build_point_id(self._get_business_id(document)),
                    vector={
                        DENSE_VECTOR_NAME: dense_vector,
                    },
                    payload=self._build_payload(document),
                )
                for document, dense_vector in zip(
                    batch,
                    dense_vectors,
                    strict=True,
                )
            ]

            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

            indexed_count += len(points)

        return IndexConstructionResult(
            collection_name=self.collection_name,
            indexed_count=indexed_count,
            embedding_dimension=embedding_dimension,
        )

    async def collection_exists(self) -> bool:
        return await self.client.collection_exists(collection_name=self.collection_name)

    async def ensure_collection(self) -> None:
        if await self.collection_exists():
            return

        vector = await self.dense_embeddings.aembed_query(
            "collection bootstrap",
        )
        if not vector:
            raise RuntimeError("dense embedding provider returned an empty vector")
        await self._ensure_collection(len(vector))

    async def _ensure_collection(
        self,
        embedding_dimension: int,
    ) -> None:
        if await self.collection_exists():
            return

        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=embedding_dimension,
                    distance=self.distance,
                )
            },
            hnsw_config=models.HnswConfigDiff(m=0, payload_m=16, ef_construct=100),
        )

        for field_name, field_schema in self._payload_indexes():
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )

    def _build_point_id(
        self,
        business_id: str,
    ) -> str:
        return str(
            uuid5(
                NAMESPACE_URL,
                (f"qdrant://{self.collection_name}/{business_id}"),
            )
        )

    @abstractmethod
    def _get_business_id(
        self,
        document: Document,
    ) -> str:
        pass

    @abstractmethod
    def _build_embedding_text(
        self,
        document: Document,
    ) -> str:
        pass

    @abstractmethod
    def _build_payload(
        self,
        document: Document,
    ) -> dict[str, Any]:
        pass

    def _payload_indexes(
        self,
    ) -> list[tuple[str, models.PayloadFieldSchema]]:
        return []

    @staticmethod
    def _chunked(
        items: list[Document],
        size: int,
    ) -> Iterator[list[Document]]:
        for index in range(0, len(items), size):
            yield items[index : index + size]
