from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
import logging
from typing import Any, Iterator
from uuid import NAMESPACE_URL, uuid5

from fastembed import SparseEmbedding, SparseTextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import AsyncQdrantClient, models

from app.config import settings

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


@lru_cache(maxsize = 1)
def get_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        url = settings.QDRANT_URL,
        api_key = settings.QDRANT_API_KEY or None,
    )


@lru_cache(maxsize = 1)
def get_dense_embeddings() -> Embeddings:
    return HuggingFaceEmbeddings(
        model_name = settings.DENSE_EMBEDDING_MODEL_NAME,
        model_kwargs = {
            "device": settings.EMBEDDING_DEVICE,
        },
        encode_kwargs = {
            "normalize_embeddings": True,
        },
    )


@lru_cache(maxsize = 1)
def get_sparse_embeddings() -> SparseTextEmbedding:
    return SparseTextEmbedding(
        model_name = settings.SPARSE_EMBEDDING_MODEL_NAME,
    )


async def close_index_resources() -> None:
    if get_qdrant_client.cache_info().currsize:
        await get_qdrant_client().close()
        get_qdrant_client.cache_clear()

    get_dense_embeddings.cache_clear()
    get_sparse_embeddings.cache_clear()


@dataclass(frozen = True, slots = True)
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
        sparse_embeddings: SparseTextEmbedding | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")

        self.collection_name = collection_name
        self.batch_size = batch_size
        self.distance = distance

        self.client = client or get_qdrant_client()
        self.dense_embeddings = dense_embeddings or get_dense_embeddings()
        self.sparse_embeddings = sparse_embeddings or get_sparse_embeddings()
        

    async def build_index(
        self,
        documents: list[Document],
    ) -> IndexConstructionResult:
        logger.info(f"Build index number: {len(documents)}")

        if not documents:
            return IndexConstructionResult(
                collection_name = self.collection_name,
                indexed_count = 0,
                embedding_dimension = 0,
            )

        indexed_count = 0
        embedding_dimension = 0

        for batch in self._chunked(
            documents,
            self.batch_size,
        ):
            texts = [
                self._build_embedding_text(document)
                for document in batch
            ]

            dense_vectors, sparse_vectors = await asyncio.gather(
                self.dense_embeddings.aembed_documents(texts),
                asyncio.to_thread(
                    self._embed_sparse_documents,
                    texts,
                ),
            )

            if embedding_dimension == 0:
                embedding_dimension = len(dense_vectors[0])

                await self._ensure_collection(embedding_dimension)

            points = [
                models.PointStruct(
                    id = self._build_point_id(
                        self._get_business_id(document)
                    ),
                    vector = {
                        DENSE_VECTOR_NAME: dense_vector,
                        SPARSE_VECTOR_NAME: sparse_vector,
                    },
                    payload = self._build_payload(document),
                )
                for document, dense_vector, sparse_vector
                in zip(
                    batch,
                    dense_vectors,
                    sparse_vectors,
                    strict = True,
                )
            ]

            await self.client.upsert(
                collection_name = self.collection_name,
                points = points,
                wait = True,
            )

            indexed_count += len(points)

        return IndexConstructionResult(
            collection_name = self.collection_name,
            indexed_count = indexed_count,
            embedding_dimension = embedding_dimension,
        )

    async def collection_exists(self) -> bool:
        return await self.client.collection_exists(
            collection_name = self.collection_name
        )

    async def _ensure_collection(
        self,
        embedding_dimension: int,
    ) -> None:
        if await self.collection_exists():
            return

        await self.client.create_collection(
            collection_name = self.collection_name,
            vectors_config = {
                DENSE_VECTOR_NAME: models.VectorParams(
                    size = embedding_dimension,
                    distance = self.distance,
                )
            },
            sparse_vectors_config = {
                SPARSE_VECTOR_NAME:
                    models.SparseVectorParams(
                        modifier = models.Modifier.IDF,
                    )
            },
        )

        for (field_name, field_schema) in self._payload_indexes():
            await self.client.create_payload_index(
                collection_name = self.collection_name,
                field_name = field_name,
                field_schema = field_schema,
                wait = True,
            )

    def _embed_sparse_documents(
        self,
        texts: list[str],
    ) -> list[models.SparseVector]:
        embeddings = list(
            self.sparse_embeddings.embed(
                texts,
                batch_size = self.batch_size,
            )
        )

        return [
            self._to_qdrant_sparse_vector(embedding)
            for embedding in embeddings
        ]

    @staticmethod
    def _to_qdrant_sparse_vector(
        embedding: SparseEmbedding,
    ) -> models.SparseVector:
        return models.SparseVector(
            indices = [
                int(index)
                for index in embedding.indices
            ],
            values = [
                float(value)
                for value in embedding.values
            ],
        )

    def _build_point_id(
        self,
        business_id: str,
    ) -> str:
        return str(
            uuid5(
                NAMESPACE_URL,
                (
                    f"qdrant://"
                    f"{self.collection_name}/"
                    f"{business_id}"
                ),
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
    ) -> list[
        tuple[str, models.PayloadSchemaType]
    ]:
        return []

    @staticmethod
    def _chunked(
        items: list[Document],
        size: int,
    ) -> Iterator[list[Document]]:
        for index in range(
            0,
            len(items),
            size,
        ):
            yield items[index:index + size]