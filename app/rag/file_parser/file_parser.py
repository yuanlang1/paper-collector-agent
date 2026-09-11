from __future__ import annotations

import asyncio
from collections import defaultdict
import logging
from pathlib import PurePosixPath
from typing import Any, TypeVar, cast
from urllib.parse import urlparse

from langchain_core.documents import Document

from app.config import settings
from app.infrastructure.mineru.mineru_client import MinerUClient
from app.schemas.file_parser import (
    BatchDocumentParseResult,
    FileParseRequest,
    FileType,
    ParsedFileDocument,
    PdfSourceBlock,
)

T = TypeVar("T")

logger = logging.getLogger(__name__)

PDF_CONTENT_BLOCK_TYPES = frozenset({
    "text",
    "title",
    "equation",
    "table",
    "image",
    "chart",
    "code",
    "algorithm",
    "list",
    "index",
})

GENERAL_FILE_TYPES: set[FileType] = {
    FileType.PDF,
    FileType.DOC,
    FileType.DOCX,
    FileType.PPT,
    FileType.PPTX,
    FileType.XLS,
    FileType.XLSX,
    FileType.PNG,
    FileType.JPG,
    FileType.JPEG,
    FileType.JP2,
    FileType.WEBP,
    FileType.GIF,
    FileType.BMP,
}


class BatchFileParseError(RuntimeError):
    def __init__(self, errors: dict[str, str]) -> None:
        self.errors = errors
        super().__init__(f"{len(errors)} files failed to parse")


class FileParser:
    MAX_BATCH_SIZE = settings.MINERU_MAX_BATCH_SIZE

    def __init__(
        self,
        *,
        mineru_client: MinerUClient,
        max_parallel_batches: int = settings.MINERU_MAX_PARALLEL_BATCHES,
    ) -> None:
        if max_parallel_batches < 1:
            raise ValueError("max_parallel_batches must be greater than 0")

        self.mineru_client = mineru_client
        self.max_parallel_batches = max_parallel_batches

    async def parse_one(
        self,
        request: FileParseRequest,
    ) -> Document:
        normalized = self._normalize_request(request)
        return (await self._parse_single(normalized)).document

    async def parse_files_result(
        self,
        requests: list[FileParseRequest],
    ) -> BatchDocumentParseResult:

        

        if not requests:
            return BatchDocumentParseResult(
                total = 0,
                documents = [],
                errors = {},
            )

        normalized = [
            self._normalize_request(request)
            for request in requests
        ]

        file_ids = [
            request.file_id
            for request in normalized
        ]
        if len(file_ids) != len(set(file_ids)):
            raise ValueError("file_id must be unique within one parse operation")

        batches = [
            batch
            for group in self._group_compatible_requests(normalized)
            for batch in self._chunked(group, self.MAX_BATCH_SIZE)
        ]

        semaphore = asyncio.Semaphore(self.max_parallel_batches)

        async def execute_batch(
            batch: list[FileParseRequest],
        ) -> tuple[list[ParsedFileDocument], dict[str, str]]:
            async with semaphore:
                return await self._parse_batch(batch)

        batch_results = await asyncio.gather(
            *(execute_batch(batch) for batch in batches),
            return_exceptions = True,
        )

        documents: list[ParsedFileDocument] = []
        errors: dict[str, str] = {}

        for batch, result in zip(
            batches,
            batch_results,
            strict = True,
        ):
            if isinstance(result, Exception):
                errors.update(
                    {
                        request.file_id: self._format_error(result)
                        for request in batch
                    }
                )
                continue

            batch_documents, batch_errors = result
            documents.extend(batch_documents)
            errors.update(batch_errors)

        documents_by_id = {
            str(parsed.document.metadata["file_id"]): parsed
            for parsed in documents
        }

        return BatchDocumentParseResult(
            total = len(normalized),
            documents = [
                documents_by_id[request.file_id]
                for request in normalized
                if request.file_id in documents_by_id
            ],
            errors = errors,
        )

    async def parse_files(
        self,
        requests: list[FileParseRequest],
    ) -> list[Document]:
        result = await self.parse_files_result(requests)

        if result.errors:
            raise BatchFileParseError(result.errors)

        return [parsed.document for parsed in result.documents]

    async def _parse_single(
        self,
        request: FileParseRequest,
    ) -> ParsedFileDocument:
        file_type = cast(FileType, request.file_type)
        model_version = self._select_model(file_type)

        task_id = await self.mineru_client.submit_single_url(
            uri = str(request.uri),
            data_id = request.file_id,
            model_version = model_version,
            options = self._build_single_options(request),
        )

        result = await self.mineru_client.wait_single(task_id)

        return await self._build_document(
            request = request,
            result = result,
            model_version = model_version,
            task_id = task_id,
        )

    async def _parse_batch(
        self,
        requests: list[FileParseRequest],
    ) -> tuple[list[ParsedFileDocument], dict[str, str]]:
        if len(requests) == 1:
            request = requests[0]

            try:
                document = await self._parse_single(request)
                return [document], {}
            except Exception as error:
                return [], {
                    request.file_id: self._format_error(error),
                }

        file_type = cast(FileType, requests[0].file_type)
        model_version = self._select_model(file_type)

        batch_id = await self.mineru_client.submit_batch_urls(
            files = [
                {
                    "url": str(request.uri),
                    "data_id": request.file_id,
                    **self._build_file_options(request),
                }
                for request in requests
            ],
            model_version = model_version,
            options = self._build_batch_options(requests[0]),
        )

        results = await self.mineru_client.wait_batch(batch_id)

        results_by_id = {
            str(result["data_id"]): result
            for result in results
            if result.get("data_id") is not None
        }

        documents: list[ParsedFileDocument] = []
        errors: dict[str, str] = {}

        for request in requests:
            result = results_by_id.get(request.file_id)

            if result is None:
                errors[request.file_id] = "MinerU batch result is missing"
                continue

            if result["state"] == "failed":
                errors[request.file_id] = result.get("err_msg") or "MinerU parsing failed"
                continue

            try:
                document = await self._build_document(
                    request = request,
                    result = result,
                    model_version = model_version,
                    batch_id = batch_id,
                )
                documents.append(document)
            except Exception as error:
                errors[request.file_id] = self._format_error(error)

        return documents, errors

    async def _build_document(
        self,
        *,
        request: FileParseRequest,
        result: dict[str, Any],
        model_version: str,
        task_id: str | None = None,
        batch_id: str | None = None,
    ) -> ParsedFileDocument:
        zip_content = await self.mineru_client.download_result_zip(
            result["full_zip_url"]
        )
        result_files = self.mineru_client.extract_result_files(zip_content)
        markdown = self.mineru_client.extract_markdown(result_files)

        file_type = cast(FileType, request.file_type)
        pdf_source_blocks = ()

        if file_type == FileType.PDF:
            content_list = self.mineru_client.extract_content_list(result_files)
            pdf_source_blocks = self._build_pdf_source_blocks(content_list)

            if not pdf_source_blocks:
                logger.warning(
                    "MinerU PDF result has no usable content_list; "
                    "falling back to generic chunking: file_id=%s",
                    request.file_id,
                )

        return ParsedFileDocument(
            document = Document(
                page_content = markdown,
                metadata = {
                    **request.metadata,
                    "file_id": request.file_id,
                    "file_name": request.file_name,
                    "file_type": file_type.value,
                    "source": str(request.uri),
                    "doc_type": "parent",
                    "parent_id": request.file_id,
                    "parser": "mineru",
                    "parser_model": model_version,
                    "mineru_task_id": task_id,
                    "mineru_batch_id": batch_id,
                },
            ),
            pdf_source_blocks = pdf_source_blocks,
        )

    @classmethod
    def _build_pdf_source_blocks(
        cls,
        content_list: list[dict[str, Any]] | None,
    ) -> tuple[PdfSourceBlock, ...]:
        if not content_list:
            return ()

        blocks = []
        for source_index, item in enumerate(content_list):
            block_type = str(item.get("type") or "").lower()
            page_idx = item.get("page_idx")

            if (
                block_type not in PDF_CONTENT_BLOCK_TYPES
                or not isinstance(page_idx, int)
                or page_idx < 0
            ):
                continue

            markdown = cls._content_item_to_markdown(item, block_type)
            if not markdown:
                continue

            bbox = item.get("bbox")
            blocks.append(
                PdfSourceBlock(
                    markdown = markdown,
                    page_no = page_idx + 1,
                    bbox = bbox if isinstance(bbox, list) else None,
                    block_type = block_type,
                    source_index = source_index,
                )
            )

        return tuple(blocks)

    @classmethod
    def _content_item_to_markdown(
        cls,
        item: dict[str, Any],
        block_type: str,
    ) -> str:
        if block_type in {"text", "title"}:
            text = cls._text_value(item.get("text") or item.get("content"))
            level = item.get("text_level")
            if text and isinstance(level, int) and level > 0:
                return f"{'#' * min(level, 4)} {text}"
            return text

        if block_type == "equation":
            return cls._text_value(item.get("text") or item.get("content"))

        if block_type in {"table", "image", "chart"}:
            return cls._join_text_values(
                item.get(f"{block_type}_caption"),
                item.get("content"),
                item.get("table_body"),
                item.get(f"{block_type}_footnote"),
            )

        if block_type in {"code", "algorithm"}:
            body = cls._text_value(item.get("code_body"))
            if body:
                body = f"```\n{body}\n```"
            return cls._join_text_values(item.get("code_caption"), body)

        if block_type in {"list", "index"}:
            items = item.get("list_items")
            if isinstance(items, list):
                return "\n".join(
                    f"- {value}"
                    for value in items
                    if isinstance(value, str) and value.strip()
                )
            return cls._text_value(item.get("text") or item.get("content"))

        return ""

    @staticmethod
    def _join_text_values(*values: Any) -> str:
        return "\n\n".join(
            text
            for value in values
            if (text := FileParser._text_value(value))
        )

    @staticmethod
    def _text_value(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return "\n".join(
                item.strip()
                for item in value
                if isinstance(item, str) and item.strip()
            )
        return ""

    def _normalize_request(
        self,
        request: FileParseRequest,
    ) -> FileParseRequest:
        file_name = request.file_name or self._filename_from_uri(str(request.uri))
        file_type = request.file_type or self._detect_file_type(file_name)

        return request.model_copy(
            update = {
                "file_name": file_name,
                "file_type": file_type,
            }
        )

    def _group_compatible_requests(
        self,
        requests: list[FileParseRequest],
    ) -> list[list[FileParseRequest]]:
        groups: dict[
            tuple[Any, ...],
            list[FileParseRequest],
        ] = defaultdict(list)

        for request in requests:
            options = request.options
            file_type = cast(FileType, request.file_type)

            key = (
                self._select_model(file_type),
                options.language,
                options.enable_formula,
                options.enable_table,
                tuple(options.extra_formats),
                options.no_cache,
                options.cache_tolerance,
            )

            groups[key].append(request)

        return list(groups.values())

    @staticmethod
    def _format_error(error: BaseException) -> str:
        message = str(error).strip()
        if message:
            return f"{type(error).__name__}: {message}"
        return f"{type(error).__name__}: {error!r}"

    @staticmethod
    def _select_model(
        file_type: FileType,
    ) -> str:
        if file_type == FileType.HTML:
            return "MinerU-HTML"

        if file_type in GENERAL_FILE_TYPES:
            return "vlm"

        raise ValueError(f"Unsupported file type: {file_type}")

    @staticmethod
    def _build_single_options(
        request: FileParseRequest,
    ) -> dict[str, Any]:
        options = request.options

        if request.file_type == FileType.HTML:
            return {
                "no_cache": options.no_cache,
                "cache_tolerance": options.cache_tolerance,
            }

        result: dict[str, Any] = {
            "is_ocr": options.is_ocr,
            "enable_formula": options.enable_formula,
            "enable_table": options.enable_table,
            "language": options.language,
            "extra_formats": options.extra_formats,
            "no_cache": options.no_cache,
            "cache_tolerance": options.cache_tolerance,
        }

        if options.page_ranges:
            result["page_ranges"] = options.page_ranges

        return result

    @staticmethod
    def _build_batch_options(
        request: FileParseRequest,
    ) -> dict[str, Any]:
        options = request.options

        if request.file_type == FileType.HTML:
            return {
                "no_cache": options.no_cache,
                "cache_tolerance": options.cache_tolerance,
            }

        return {
            "enable_formula": options.enable_formula,
            "enable_table": options.enable_table,
            "language": options.language,
            "extra_formats": options.extra_formats,
            "no_cache": options.no_cache,
            "cache_tolerance": options.cache_tolerance,
        }

    @staticmethod
    def _build_file_options(
        request: FileParseRequest,
    ) -> dict[str, Any]:
        if request.file_type == FileType.HTML:
            return {}

        result: dict[str, Any] = {
            "is_ocr": request.options.is_ocr,
        }

        if request.options.page_ranges:
            result["page_ranges"] = request.options.page_ranges

        return result

    @staticmethod
    def _filename_from_uri(
        uri: str,
    ) -> str:
        file_name = PurePosixPath(
            urlparse(uri).path
        ).name

        if not file_name:
            raise ValueError("Cannot determine file name from URI; provide file_name explicitly")

        return file_name

    @staticmethod
    def _detect_file_type(
        file_name: str,
    ) -> FileType:
        suffix = PurePosixPath(file_name).suffix.lower()

        try:
            return FileType(suffix.removeprefix("."))
        except ValueError as error:
            raise ValueError(
                f"Unsupported file type: "
                f"{suffix or file_name}"
            ) from error

    @staticmethod
    def _chunked(
        items: list[T],
        size: int,
    ) -> list[list[T]]:
        return [
            items[index:index + size]
            for index in range(
                0,
                len(items),
                size,
            )
        ]
