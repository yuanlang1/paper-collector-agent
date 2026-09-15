from __future__ import annotations

import hashlib
import json
from typing import Any

from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient, models

from app.config import settings
from app.rag.index_construction.base import get_qdrant_client
from app.rag.retrieval.paper_content_retrieval import (
    paper_content_document_from_payload,
)


class PaperContentCorpusReader:
    def __init__(
        self,
        *,
        collection_name: str = settings.PAPER_CONTENT_COLLECTION_NAME,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.client = client or get_qdrant_client()

    async def read(self, paper_ids: list[str]) -> list[Document]:
        normalized_ids = sorted({str(paper_id) for paper_id in paper_ids})
        if not normalized_ids:
            return []

        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="paper_id",
                    match=models.MatchAny(any=normalized_ids),
                )
            ]
        )
        offset = None
        documents: list[Document] = []
        while True:
            records, offset = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                offset=offset,
                limit=256,
                with_payload=True,
                with_vectors=False,
            )
            documents.extend(
                paper_content_document_from_payload(record.payload or {})
                for record in records
            )
            if offset is None:
                break

        return sorted(
            documents,
            key=lambda document: (
                str(document.metadata["paper_id"]),
                int(document.metadata["chunk_index"]),
            ),
        )

    @staticmethod
    def page_aware_documents(documents: list[Document]) -> list[Document]:
        return [
            document
            for document in documents
            if PaperContentCorpusReader.is_page_aware(document)
        ]

    @staticmethod
    def is_page_aware(document: Document) -> bool:
        page_numbers = document.metadata.get("page_numbers")
        spans = document.metadata.get("source_spans")
        if not isinstance(page_numbers, list) or not page_numbers:
            return False
        if not isinstance(spans, list) or not spans:
            return False

        return all(
            isinstance(span, dict)
            and isinstance(span.get("page"), int)
            and isinstance(span.get("char_start"), int)
            and isinstance(span.get("char_end"), int)
            and span["char_start"] < span["char_end"]
            for span in spans
        )

    @staticmethod
    def fingerprint(documents: list[Document]) -> str:
        digest = hashlib.sha256()
        for document in sorted(
            documents,
            key=lambda item: (
                str(item.metadata["paper_id"]),
                int(item.metadata["chunk_index"]),
            ),
        ):
            value: dict[str, Any] = {
                "paper_id": str(document.metadata["paper_id"]),
                "chunk_id": str(document.metadata["chunk_id"]),
                "content_sha256": hashlib.sha256(
                    document.page_content.encode("utf-8")
                ).hexdigest(),
                "page_numbers": document.metadata.get("page_numbers", []),
                "source_spans": document.metadata.get("source_spans", []),
            }
            digest.update(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        return digest.hexdigest()
