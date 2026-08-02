from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import AsyncQdrantClient, models

from app.rag.index_construction.base import BaseQdrantIndexConstructionModule, IndexConstructionResult

logger = logging.getLogger(__name__)

class PaperContentIndexConstructionModule(
    BaseQdrantIndexConstructionModule
):
    def __init__(
        self,
        *,
        collection_name: str = "paper_content",
        batch_size: int = 32,
    ) -> None:
        super().__init__(
            collection_name = collection_name,
            batch_size = batch_size,
            distance = models.Distance.COSINE,
        )

    async def build_index(
        self,
        documents: list[Document],
        *,
        replace_existing: bool = True,
    ) -> IndexConstructionResult:    
        result = await super().build_index(documents)

        if replace_existing and documents:
            chunk_ids_by_paper: dict[str, set[str]] = {}

            for document in documents:
                paper_id = str(document.metadata["paper_id"])
                chunk_ids_by_paper.setdefault(paper_id, set()).add(
                    str(document.metadata["chunk_id"])
                )

            for paper_id, chunk_ids in chunk_ids_by_paper.items():
                await self._delete_stale_paper_contents(
                    paper_id,
                    chunk_ids,
                )

        return result

    def _get_business_id(
        self,
        document: Document,
    ) -> str:
        return str(document.metadata["chunk_id"])

    def _build_embedding_text(
        self,
        document: Document,
    ) -> str:
        return document.page_content.strip()

    def _build_payload(
        self,
        document: Document,
    ) -> dict[str, Any]:
        metadata = document.metadata
        paper_id = str(metadata["paper_id"])

        payload: dict[str, Any] = {
            "paper_id": paper_id,
            "parent_id": str(
                metadata.get("parent_id")
                or paper_id
            ),
            "chunk_id": str(metadata["chunk_id"]),
            "chunk_index": int(metadata["chunk_index"]),
            "content": document.page_content,
            "section_path": str(metadata.get("section_path") or "")
        }

        if metadata.get("year") is not None:
            payload["year"] = int(metadata["year"])

        return payload

    def _payload_indexes(
        self,
    ) -> list[
        tuple[str, models.PayloadSchemaType]
    ]:
        return [
            (
                "paper_id",
                models.PayloadSchemaType.KEYWORD,
            ),
            (
                "parent_id",
                models.PayloadSchemaType.KEYWORD,
            ),
        ]

    async def delete_paper_contents(
        self,
        paper_ids: list[str],
    ) -> None:
        if not paper_ids:
            return

        exists = await self.client.collection_exists(collection_name = self.collection_name)

        if not exists:
            return

        await self.client.delete(
            collection_name = self.collection_name,
            points_selector = models.FilterSelector(
                filter = models.Filter(
                    must = [
                        models.FieldCondition(
                            key = "paper_id",
                            match = models.MatchAny(
                                any = [
                                    str(paper_id)
                                    for paper_id in paper_ids
                                ]
                            ),
                        )
                    ]
                )
            ),
            wait = True,
        )
    
    async def _delete_stale_paper_contents(
        self,
        paper_id: str,
        retained_chunk_ids: set[str],
    ) -> None:
        await self.client.delete(
            collection_name = self.collection_name,
            points_selector = models.FilterSelector(
                filter = models.Filter(
                    must = [
                        models.FieldCondition(
                            key = "paper_id",
                            match = models.MatchValue(value = paper_id),
                        )
                    ],
                    must_not = [
                        models.FieldCondition(
                            key = "chunk_id",
                            match = models.MatchAny(
                                any = sorted(retained_chunk_ids)
                            ),
                        )
                    ],
                )
            ),
            wait = True,
        )