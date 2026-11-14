import logging
from dataclasses import dataclass
from typing import Literal

from langchain_core.documents import Document

from app.infrastructure.oss import OssObjectStore
from app.infrastructure.oss import oss_store as shared_oss_store
from app.rag.data_preparation.data_preparation import DataPreparationModule
from app.rag.index_construction.paper_content_index_construction import (
    PaperContentIndexConstructionModule,
)
from app.rag.index_construction.paper_index_construction import (
    PaperIndexConstructionModule,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PaperRagInput:
    paper_id: int
    user_id: str
    title: str
    abstract: str
    pdf_url: str = ""
    oss_name: str = ""
    year: int | None = None


@dataclass(frozen=True, slots=True)
class PaperRagResult:
    paper_id: int
    status: Literal["ready", "skipped", "failed"]
    chunk_count: int = 0
    error: str | None = None


class PaperRagProcessor:
    def __init__(
        self,
        *,
        data_preparation: DataPreparationModule,
        paper_index: PaperIndexConstructionModule | None = None,
        paper_content_index: PaperContentIndexConstructionModule | None = None,
        oss_store: OssObjectStore | None = None,
    ) -> None:
        self.data_preparation = data_preparation
        self.paper_index = paper_index
        self.paper_content_index = paper_content_index
        self.oss_store = oss_store or shared_oss_store

    async def process_many(
        self,
        papers: list[PaperRagInput],
    ) -> list[PaperRagResult]:
        paper_ids = {paper.paper_id for paper in papers}
        if len(paper_ids) != len(papers):
            raise ValueError("paper_id must be unique in one RAG batch")

        # ponytail: Index one paper at a time to bound memory. Overlap OSS reads if needed.
        return [await self._index_paper(paper) for paper in papers]

    async def _index_paper(self, paper: PaperRagInput) -> PaperRagResult:
        oss_name = paper.oss_name.strip()
        if not oss_name:
            return PaperRagResult(
                paper_id=paper.paper_id, status="failed", error="missing OSS name"
            )

        try:
            logger.info("paper: %s start RAG process.", paper.paper_id)
            markdown = await self.oss_store.get_paper_markdown(oss_name)
            document = Document(
                page_content=markdown.content,
                metadata={
                    "user_id": paper.user_id,
                    "paper_id": str(paper.paper_id),
                    "parent_id": str(paper.paper_id),
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "year": paper.year,
                    "source": paper.pdf_url or f"oss://{markdown.object_name}",
                },
            )
            chunks = self.data_preparation.prepare_documents([document])
            for chunk in chunks:
                chunk.metadata.pop("section_path", None)

            if self.paper_index is None:
                self.paper_index = PaperIndexConstructionModule()
            if self.paper_content_index is None:
                self.paper_content_index = PaperContentIndexConstructionModule()

            await self.paper_index.build_index([document])
            if not chunks:
                await self.paper_content_index.delete_paper_contents(
                    [str(paper.paper_id)], user_id=paper.user_id
                )
                return PaperRagResult(
                    paper_id=paper.paper_id,
                    status="skipped",
                    error="no indexable document content",
                )

            await self.paper_content_index.build_index(chunks)
            return PaperRagResult(
                paper_id=paper.paper_id,
                status="ready",
                chunk_count=len(chunks),
            )

        except Exception as error:
            return PaperRagResult(
                paper_id=paper.paper_id,
                status="failed",
                error=str(error),
            )
