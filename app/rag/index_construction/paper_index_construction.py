from __future__ import annotations

from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient, models
from langchain_core.embeddings import Embeddings

from app.config import settings
from app.rag.index_construction.base import BaseQdrantIndexConstructionModule

class PaperIndexConstructionModule(
    BaseQdrantIndexConstructionModule
):
    def __init__(
        self,
        *,
        collection_name: str = settings.PAPER_COLLECTION_NAME,
        batch_size: int = 32,
    ) -> None:
        super().__init__(
            collection_name = collection_name,
            batch_size = batch_size,
            distance = models.Distance.COSINE,
        )

    def _get_business_id(
        self,
        document: Document,
    ) -> str:
        return str(document.metadata["paper_id"])

    def _build_embedding_text(
        self,
        document: Document,
    ) -> str:
        title = str(document.metadata.get("title") or "").strip()
        abstract = str(document.metadata.get("abstract") or "").strip()

        return "\n\n".join([title, abstract])

    def _build_payload(
        self,
        document: Document,
    ) -> dict[str, str]:
        metadata = document.metadata

        return {
            "paper_id": str(metadata["paper_id"]),
            "title": str(metadata.get("title") or ""),
            "abstract": str(metadata.get("abstract") or ""),
        }

    def _payload_indexes(
        self,
    ) -> list[
        tuple[str, models.PayloadSchemaType]
    ]:
        return [
            (
                "paper_id",
                models.PayloadSchemaType.KEYWORD,
            )
        ]

    async def delete_papers(
        self,
        paper_ids: list[str],
    ) -> None:
        if not paper_ids:
            return

        exists = await self.client.collection_exists(collection_name = self.collection_name)

        if not exists:
            return

        point_ids = [
            self._build_point_id(
                str(paper_id)
            )
            for paper_id in paper_ids
        ]

        await self.client.delete(
            collection_name = self.collection_name,
            points_selector = models.PointIdsList(
                points = point_ids
            ),
            wait = True,
        )