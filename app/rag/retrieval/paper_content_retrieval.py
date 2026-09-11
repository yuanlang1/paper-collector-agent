from __future__ import annotations

from langchain_core.documents import Document
from qdrant_client import models

from app.config import settings
from app.rag.retrieval.base import BaseHybridRetrievalModule

class PaperContentHybridRetrievalModule(
    BaseHybridRetrievalModule
):
    def __init__(
        self,
        *,
        collection_name: str = settings.PAPER_CONTENT_COLLECTION_NAME,
        default_top_k: int = 20,
    ) -> None:
        super().__init__(
            collection_name = collection_name,
            default_top_k = default_top_k,
            default_prefetch_limit = 80,
        )

    async def search(
        self,
        query: str,
        *,
        sparse_query: str | None = None,
        top_k: int | None = None,
        paper_ids: list[str] | None = None,
    ) -> list[Document]:
        if paper_ids is not None:
            if not paper_ids:
                return []

            query_filter = models.Filter(
                must = [
                    models.FieldCondition(
                        key = "paper_id",
                        match = models.MatchAny(
                            any = paper_ids
                        ),
                    )
                ]
            )
        else:
            query_filter = None

        return await super().search(
            query,
            sparse_query = sparse_query,
            top_k = top_k,
            query_filter = query_filter,
        )

    def _payload_to_document(
        self,
        payload: dict,
    ) -> Document:
        metadata = {
            "paper_id": str(payload["paper_id"]),
            "chunk_id": str(payload["chunk_id"]),
            "chunk_index": int(payload["chunk_index"]),
            "section_path": str(payload.get("section_path") or ""),
            "retrieval_type": "paper_content",
        }

        if payload.get("page_start") is not None:
            metadata.update(
                {
                    "page_start": int(payload["page_start"]),
                    "page_end": int(payload["page_end"]),
                    "page_numbers": [
                        int(page)
                        for page in payload.get("page_numbers", [])
                    ],
                    "source_spans": list(
                        payload.get("source_spans", [])
                    ),
                }
            )

        return Document(
            page_content = str(payload.get("content") or ""),
            metadata = metadata,
        )
