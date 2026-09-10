from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient, models

from app.config import settings
from app.models.memory import MemoryFact
from app.rag.index_construction.base import BaseQdrantIndexConstructionModule


class MemoryFactIndexConstructionModule(BaseQdrantIndexConstructionModule):
    def __init__(
        self,
        *,
        collection_name: str = settings.MEMORY_FACT_COLLECTION_NAME,
        batch_size: int = 32,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        super().__init__(
            collection_name=collection_name,
            batch_size=batch_size,
            client=client,
        )

    async def upsert_facts(self, facts: Sequence[MemoryFact]) -> None:
        await self.build_index([self._to_document(fact) for fact in facts])

    def _get_business_id(self, document: Document) -> str:
        return str(document.metadata["fact_id"])

    def _build_embedding_text(self, document: Document) -> str:
        return document.page_content

    def _build_payload(self, document: Document) -> dict[str, object]:
        return document.metadata

    def _payload_indexes(self) -> list[tuple[str, models.PayloadSchemaType]]:
        return [
            ("fact_id", models.PayloadSchemaType.INTEGER),
            ("user_id", models.PayloadSchemaType.KEYWORD),
            ("status", models.PayloadSchemaType.KEYWORD),
        ]

    @staticmethod
    def _to_document(fact: MemoryFact) -> Document:
        return Document(
            page_content=f"{fact.subject}\n{fact.content}",
            metadata={
                "fact_id": fact.id,
                "user_id": fact.user_id,
                "status": fact.status,
                "source": fact.source,
            },
        )
