from __future__ import annotations

from langchain_core.documents import Document
from qdrant_client import models

from app.config import settings
from app.rag.retrieval.base import BaseHybridRetrievalModule

class PaperHybridRetrievalModule(
    BaseHybridRetrievalModule
):
    def __init__(
        self,
        *,
        collection_name: str = settings.PAPER_COLLECTION_NAME,
        default_top_k: int = 10,
    ) -> None:
        super().__init__(
            collection_name = collection_name,
            default_top_k = default_top_k,
            default_prefetch_limit = 50,
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
                        match = models.MatchAny(any = paper_ids),
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
        title = str(payload.get("title") or "")
        abstract = str(payload.get("abstract") or "")

        return Document(
            page_content = "\n\n".join(
                part
                for part in (
                    title,
                    abstract,
                )
                if part
            ),
            metadata = {
                "paper_id": str(payload["paper_id"]),
                "title": title,
                "abstract": abstract,
                "retrieval_type": "paper",
            },
        )