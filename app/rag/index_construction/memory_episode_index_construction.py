from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient, models

from app.config import settings
from app.models.memory import MemoryEpisode
from app.rag.index_construction.base import BaseQdrantIndexConstructionModule


class MemoryEpisodeIndexConstructionModule(BaseQdrantIndexConstructionModule):
    def __init__(
        self,
        *,
        collection_name: str = settings.MEMORY_EPISODE_COLLECTION_NAME,
        batch_size: int = 32,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        super().__init__(
            collection_name=collection_name,
            batch_size=batch_size,
            client=client,
        )

    async def upsert_episodes(self, episodes: Sequence[MemoryEpisode]) -> None:
        await self.build_index([self._to_document(episode) for episode in episodes])

    def _get_business_id(self, document: Document) -> str:
        return str(document.metadata["episode_id"])

    def _build_embedding_text(self, document: Document) -> str:
        return document.page_content

    def _build_payload(self, document: Document) -> dict[str, object]:
        return document.metadata

    def _payload_indexes(self) -> list[tuple[str, models.PayloadSchemaType]]:
        return [
            ("episode_id", models.PayloadSchemaType.INTEGER),
            ("user_id", models.PayloadSchemaType.KEYWORD),
            ("conversation_id", models.PayloadSchemaType.KEYWORD),
            ("status", models.PayloadSchemaType.KEYWORD),
        ]

    @staticmethod
    def _to_document(episode: MemoryEpisode) -> Document:
        return Document(
            page_content=episode.summary,
            metadata={
                "episode_id": episode.id,
                "user_id": episode.user_id,
                "conversation_id": episode.source_conversation_id,
                "status": episode.status,
                "happened_at": episode.happened_at.isoformat(),
            },
        )
