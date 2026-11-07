from __future__ import annotations

import json

from langchain_core.documents import Document
from qdrant_client import models

from app.config import settings
from app.rag.index_construction.base import (
    BaseQdrantIndexConstructionModule,
    user_filter,
)


class PaperIndexConstructionModule(BaseQdrantIndexConstructionModule):
    def __init__(
        self,
        *,
        collection_name: str = settings.PAPER_COLLECTION_NAME,
        batch_size: int = 32,
    ) -> None:
        super().__init__(
            collection_name=collection_name,
            batch_size=batch_size,
            distance=models.Distance.COSINE,
        )

    def _get_business_id(
        self,
        document: Document,
    ) -> str:
        return json.dumps(
            [str(document.metadata["user_id"]), str(document.metadata["paper_id"])]
        )

    def _build_embedding_text(
        self,
        document: Document,
    ) -> str:
        title = str(document.metadata.get("title") or "").strip()
        abstract = str(document.metadata.get("abstract") or "").strip()

        return f"{title}\n\n{abstract}"

    def _build_payload(
        self,
        document: Document,
    ) -> dict[str, str]:
        metadata = document.metadata

        return {
            "user_id": str(metadata["user_id"]),
            "search_text": self._build_embedding_text(document),
            "paper_id": str(metadata["paper_id"]),
            "title": str(metadata.get("title") or ""),
            "abstract": str(metadata.get("abstract") or ""),
        }

    def _payload_indexes(
        self,
    ) -> list[tuple[str, models.PayloadFieldSchema]]:
        return [
            (
                "user_id",
                models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    is_tenant=True,
                ),
            ),
            (
                "search_text",
                models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.MULTILINGUAL,
                    lowercase=True,
                    scoring=models.TextScoringParams(type=models.TextScoringType.BM25),
                    enable_hnsw=False,
                ),
            ),
            (
                "paper_id",
                models.PayloadSchemaType.KEYWORD,
            ),
        ]

    async def delete_papers(
        self,
        paper_ids: list[str],
        *,
        user_id: str,
    ) -> None:
        query_filter = user_filter(user_id)
        if not paper_ids or not await self.collection_exists():
            return
        query_filter.must.append(
            models.FieldCondition(
                key="paper_id",
                match=models.MatchAny(any=paper_ids),
            )
        )
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=query_filter),
            wait=True,
        )
