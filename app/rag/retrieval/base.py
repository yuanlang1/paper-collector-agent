from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

from fastembed import SparseEmbedding, SparseTextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import AsyncQdrantClient, models

from app.rag.index_construction.base import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, get_dense_embeddings, get_qdrant_client, get_sparse_embeddings


@dataclass(frozen = True, slots = True)
class HybridQueryVectors:
    dense: list[float]
    sparse: models.SparseVector


class BaseHybridRetrievalModule(ABC):

    def __init__(
        self,
        *,
        collection_name: str,
        default_top_k: int = 10,
        default_prefetch_limit: int = 50,
        client: AsyncQdrantClient | None = None,
        dense_embeddings: Embeddings | None = None,
        sparse_embeddings: SparseTextEmbedding | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.default_top_k = default_top_k
        self.default_prefetch_limit = default_prefetch_limit

        self.client = client or get_qdrant_client()
        self.dense_embeddings = dense_embeddings or get_dense_embeddings()
        self.sparse_embeddings = sparse_embeddings or get_sparse_embeddings()
        
    async def search(
        self,
        query: str,
        *,  
        sparse_query: str | None = None,
        top_k: int | None = None,
        prefetch_limit: int | None = None,
        query_filter: models.Filter | None = None,
        dense_score_threshold: float | None = None,
    ) -> list[Document]:
        vectors = await self.encode_query(
            query,
            sparse_query = sparse_query,
        )
        return await self.search_with_vectors(
            vectors,
            top_k = top_k,
            prefetch_limit = prefetch_limit,
            query_filter = query_filter,
            dense_score_threshold = dense_score_threshold,
        )

    async def encode_query(
        self,
        query: str,
        *,
        sparse_query: str | None = None,
    ) -> HybridQueryVectors:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query cannot be empty")

        normalized_sparse_query = (
            sparse_query.strip()
            if sparse_query
            else normalized_query
        )
        dense_vector, sparse_vector = await asyncio.gather(
            self.dense_embeddings.aembed_query(normalized_query),
            asyncio.to_thread(
                self._embed_sparse_query,
                normalized_sparse_query,
            ),
        )
        return HybridQueryVectors(
            dense = dense_vector,
            sparse = sparse_vector,
        )

    async def search_with_vectors(
        self,
        vectors: HybridQueryVectors,
        *,
        top_k: int | None = None,
        prefetch_limit: int | None = None,
        query_filter: models.Filter | None = None,
        dense_score_threshold: float | None = None,
    ) -> list[Document]:
        result_limit = top_k or self.default_top_k
        candidate_limit = prefetch_limit or self.default_prefetch_limit

        response = await self.client.query_points(
            collection_name = self.collection_name,
            prefetch = [
                models.Prefetch(
                    query = vectors.dense,
                    using = DENSE_VECTOR_NAME,
                    limit = candidate_limit,
                    filter = query_filter,
                    score_threshold = (
                        dense_score_threshold
                    ),
                ),
                models.Prefetch(
                    query = vectors.sparse,
                    using = SPARSE_VECTOR_NAME,
                    filter = query_filter,
                    limit = candidate_limit,
                ),
            ],
            query = models.RrfQuery(
                rrf = models.Rrf()
            ),
            query_filter = query_filter,
            limit = result_limit,
            with_payload = True,
            with_vectors = False,
        )

        documents: list[Document] = []

        for point in response.points:
            payload = point.payload or {}
            document = self._payload_to_document(payload)

            document.metadata.update(
                {
                    "retrieval_score": float(point.score),
                    "qdrant_point_id": str(point.id),
                    "retrieval_mode": "hybrid_rrf",
                    "collection_name": self.collection_name,
                }
            )

            documents.append(document)

        return documents

    def _embed_sparse_query(
        self,
        query: str,
    ) -> models.SparseVector:
        embedding = next(
            iter(
                self.sparse_embeddings.query_embed(
                    query
                )
            )
        )

        return self._to_qdrant_sparse_vector(embedding)

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

    @abstractmethod
    def _payload_to_document(
        self,
        payload: dict,
    ) -> Document:
        pass
