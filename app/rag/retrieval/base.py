from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import AsyncQdrantClient, models

from app.rag.index_construction.base import (
    DENSE_VECTOR_NAME,
    get_dense_embeddings,
    get_qdrant_client,
    user_filter,
)


class BaseHybridRetrievalModule(ABC):
    def __init__(
        self,
        *,
        collection_name: str,
        text_field: str,
        default_top_k: int = 10,
        default_prefetch_limit: int = 50,
        client: AsyncQdrantClient | None = None,
        dense_embeddings: Embeddings | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.text_field = text_field
        self.default_top_k = default_top_k
        self.default_prefetch_limit = default_prefetch_limit
        self.client = client or get_qdrant_client()
        self.dense_embeddings = dense_embeddings or get_dense_embeddings()

    async def search(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int | None = None,
        prefetch_limit: int | None = None,
        query_filter: models.Filter | None = None,
        dense_score_threshold: float | None = None,
    ) -> list[Document]:
        tenant_filter = user_filter(user_id)
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")

        scoped_filter = models.Filter(
            must=[tenant_filter, query_filter]
            if query_filter is not None
            else [tenant_filter],
        )
        dense_vector = await self.dense_embeddings.aembed_query(query)
        candidate_limit = prefetch_limit or self.default_prefetch_limit
        response = await self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=DENSE_VECTOR_NAME,
                    limit=candidate_limit,
                    filter=scoped_filter,
                    score_threshold=dense_score_threshold,
                ),
                models.Prefetch(
                    query=models.TextQuery(text=query),
                    using=self.text_field,
                    filter=scoped_filter,
                    limit=candidate_limit,
                    params=models.SearchParams(
                        idf=models.IdfCorpusParams(corpus=tenant_filter),
                    ),
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf()),
            query_filter=scoped_filter,
            limit=top_k or self.default_top_k,
            with_payload=True,
            with_vectors=False,
        )

        documents: list[Document] = []
        for point in response.points:
            document = self._payload_to_document(point.payload or {})
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

    @abstractmethod
    def _payload_to_document(self, payload: dict) -> Document:
        pass
