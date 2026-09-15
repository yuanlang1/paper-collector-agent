import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import inspect
import logging
from typing import Literal

from langchain_core.documents import Document
from pydantic import ValidationError

from app.infrastructure.oss import (
    AliyunOssObjectStore,
    OssObjectStore,
    normalize_pdf_sha256,
)
from app.rag.data_preparation.data_preparation import DataPreparationModule
from app.rag.file_parser.file_parser import FileParser
from app.rag.index_construction.paper_content_index_construction import PaperContentIndexConstructionModule
from app.rag.index_construction.paper_index_construction import PaperIndexConstructionModule
from app.schemas.file_parser import FileParseRequest, ParsedFileDocument

logger = logging.getLogger(__name__)


@dataclass(frozen = True, slots = True)
class PaperRagInput:
    paper_id: int
    title: str
    abstract: str
    pdf_url: str
    oss_name: str = ""
    year: int | None = None

@dataclass(frozen = True, slots = True)
class PaperRagResult:
    paper_id: int
    status: Literal["ready", "skipped", "failed"]
    chunk_count: int = 0
    error: str | None = None


PaperResultListener = Callable[[PaperRagResult], Awaitable[None] | None]

class PaperRagProcessor:
    def __init__(
        self,
        *,
        file_parser: FileParser,
        data_preparation: DataPreparationModule,
        paper_index: PaperIndexConstructionModule,
        paper_content_index: PaperContentIndexConstructionModule,
        oss_store: OssObjectStore | None = None,
    ) -> None:
        self.file_parser = file_parser
        self.data_preparation = data_preparation
        self.paper_index = paper_index
        self.paper_content_index = paper_content_index
        self.oss_store = oss_store or AliyunOssObjectStore()

        self._index_lock = asyncio.Lock()

    async def process_many(
        self,
        papers: list[PaperRagInput],
        *,
        on_paper_result: PaperResultListener | None = None,
    ) -> list[PaperRagResult]:
        paper_ids = [paper.paper_id for paper in papers]

        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("paper_id must be unique in one RAG batch")

        requests: list[FileParseRequest] = []
        papers_by_file_id: dict[str, PaperRagInput] = {}
        results_by_paper_id: dict[int, PaperRagResult] = {}

        for paper in papers:
            paper_id = paper.paper_id

            oss_name = normalize_pdf_sha256(paper.oss_name)
            if not paper.pdf_url.strip() and oss_name is None:
                result = PaperRagResult(
                    paper_id = paper_id,
                    status = "skipped",
                    error = "missing PDF source",
                )
                results_by_paper_id[paper_id] = result
                await self._notify_paper_result(on_paper_result, result)
                continue
            
            file_id = f"paper-{paper_id}"
            uri = await self._source_uri(paper, oss_name)
            if not uri:
                result = PaperRagResult(
                    paper_id = paper_id,
                    status = "skipped",
                    error = "missing readable PDF source",
                )
                results_by_paper_id[paper_id] = result
                await self._notify_paper_result(on_paper_result, result)
                continue

            try:
                request = FileParseRequest(
                    file_id = file_id,
                    uri = uri,
                    file_name = f"{file_id}.pdf",
                    metadata = {
                        "source": paper.pdf_url.strip() or f"oss://{oss_name}",
                    },
                )
            except ValidationError as error:
                result = PaperRagResult(
                    paper_id = paper.paper_id,
                    status = "failed",
                    error = str(error),
                )
                results_by_paper_id[paper.paper_id] = result
                await self._notify_paper_result(on_paper_result, result)
                continue
            
            requests.append(request)
            papers_by_file_id[file_id] = paper

        parse_result = await self.file_parser.parse_files_result(requests)

        for file_id, error in parse_result.errors.items():
            paper = papers_by_file_id[file_id]
            result = PaperRagResult(
                paper_id = paper.paper_id,
                status = "failed",
                error = error,
            )
            results_by_paper_id[paper.paper_id] = result
            await self._notify_paper_result(on_paper_result, result)

        parsed_documents = [
            self._as_parsed_file_document(document)
            for document in parse_result.documents
        ]
        index_tasks = [
            asyncio.create_task(
                self._index_parsed_document(
                    papers_by_file_id[str(parsed.document.metadata["file_id"])],
                    parsed,
                )
            )
            for parsed in parsed_documents
        ]

        for task in asyncio.as_completed(index_tasks):
            result = await task
            results_by_paper_id[result.paper_id] = result
            await self._notify_paper_result(on_paper_result, result)

        for paper in papers:
            if paper.paper_id not in results_by_paper_id:
                result = PaperRagResult(
                    paper_id = paper.paper_id,
                    status = "failed",
                    error = "parser returned no document or error",
                )
                results_by_paper_id[paper.paper_id] = result
                await self._notify_paper_result(on_paper_result, result)
        
        return [
            results_by_paper_id[paper.paper_id]
            for paper in papers
        ]

    @staticmethod
    async def _notify_paper_result(
        listener: PaperResultListener | None,
        result: PaperRagResult,
    ) -> None:
        if listener is None:
            return

        try:
            value = listener(result)
            if inspect.isawaitable(value):
                await value
        except Exception:
            logger.warning(
                "Paper RAG progress listener failed: paper_id=%s",
                result.paper_id,
                exc_info=True,
            )
    
    async def _index_parsed_document(
        self,
        paper: PaperRagInput,
        parsed_document: ParsedFileDocument,
    ) -> PaperRagResult:
        try:
            logger.info("paper: %s start RAG process.", paper.paper_id)

            parent_document = self._bind_paper_metadata(
                parsed_document.document,
                paper,
            )
            await self._save_markdown(paper, parent_document.page_content)
            chunks = (
                self.data_preparation.prepare_pdf_document(
                    parent_document,
                    parsed_document.pdf_source_blocks,
                )
                if parsed_document.pdf_source_blocks
                else self.data_preparation.prepare_documents([parent_document])
            )

            async with self._index_lock:
                await self.paper_index.build_index([parent_document])

                if chunks:
                    await self.paper_content_index.build_index(chunks)

            if not chunks:
                return PaperRagResult(
                    paper_id = paper.paper_id,
                    status = "skipped",
                    error = "no indexable document content",
                )
            
            return PaperRagResult(
                paper_id = paper.paper_id,
                status = "ready",
                chunk_count = len(chunks),
            )

        except Exception as error:
            return PaperRagResult(
                paper_id = paper.paper_id,
                status = "failed",
                error = str(error),
            )

    async def _source_uri(
        self,
        paper: PaperRagInput,
        oss_name: str | None,
    ) -> str:
        fallback = paper.pdf_url.strip()
        if oss_name is None:
            return fallback

        try:
            return (
                await self.oss_store.get_pdf_temporary_url(
                    pdf_sha256=oss_name,
                )
                or fallback
            )
        except Exception:
            logger.warning(
                "Unable to read PDF from OSS; using source URL: paper_id=%s",
                paper.paper_id,
                exc_info=True,
            )
            return fallback

    async def _save_markdown(
        self,
        paper: PaperRagInput,
        markdown: str,
    ) -> None:
        oss_name = normalize_pdf_sha256(paper.oss_name)
        if oss_name is None or not markdown.strip():
            return

        try:
            await self.oss_store.upload_markdown(
                markdown=markdown,
                pdf_sha256=oss_name,
            )
        except Exception:
            logger.warning(
                "Unable to save parsed Markdown to OSS: paper_id=%s",
                paper.paper_id,
                exc_info=True,
            )

    @staticmethod
    def _as_parsed_file_document(
        value: ParsedFileDocument | Document,
    ) -> ParsedFileDocument:
        if isinstance(value, ParsedFileDocument):
            return value
        if isinstance(value, Document):
            return ParsedFileDocument(document = value)
        raise TypeError(f"Unexpected parsed document type: {type(value)!r}")

    
    @staticmethod
    def _bind_paper_metadata(
        parsed_document: Document,
        paper: PaperRagInput,
    ) -> Document:
        metadata = {
            **parsed_document.metadata,
            "paper_id": str(paper.paper_id),
            "parent_id": str(paper.paper_id),
            "title": paper.title,
            "abstract": paper.abstract,
        }

        if paper.year is not None:
            metadata["year"] = paper.year

        return Document(
            page_content = parsed_document.page_content,
            metadata = metadata,
        )
