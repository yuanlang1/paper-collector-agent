from __future__ import annotations

from langchain_core.documents import Document
from qdrant_client import models

from app.config import settings
from app.rag.retrieval.base import BaseHybridRetrievalModule, HybridQueryVectors


class MemoryEpisodeHybridRetrievalModule(BaseHybridRetrievalModule):
    def __init__(
        self,
        *,
        collection_name: str = settings.MEMORY_EPISODE_COLLECTION_NAME,
    ) -> None:
        super().__init__(
            collection_name=collection_name,
            default_top_k=6,
            default_prefetch_limit=18,
        )

    async def search_for_conversation(
        self,
        vectors: HybridQueryVectors,
        *,
        user_id: str,
        conversation_id: str,
        top_k: int = 6,
    ) -> list[Document]:
        return await self.search_with_vectors(
            vectors,
            top_k=top_k,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    ),
                    models.FieldCondition(
                        key="conversation_id",
                        match=models.MatchValue(value=conversation_id),
                    ),
                    models.FieldCondition(
                        key="status",
                        match=models.MatchValue(value="active"),
                    ),
                ]
            ),
        )

    def _payload_to_document(self, payload: dict) -> Document:
        return Document(
            page_content="",
            metadata={"episode_id": int(payload["episode_id"])},
        )
