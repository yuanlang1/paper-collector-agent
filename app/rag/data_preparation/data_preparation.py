
import hashlib
import logging
import re
from dataclasses import replace
from typing import Any, Sequence
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from app.schemas.file_parser import PdfSourceBlock

logger = logging.getLogger(__name__)

HEADERS_TO_SPLIT_ON = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
    ("####", "header_4"),
]

HEADER_KEYS = (
    "header_1",
    "header_2",
    "header_3",
    "header_4",
)

TEXT_SEPARATOR = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    ". ",
    "! ",
    "? ",
    " ",
    "",
]

PDF_ATOMIC_BLOCK_TYPES = frozenset({
    "equation",
    "table",
    "image",
    "chart",
    "code",
    "algorithm",
})

class DataPreparationModule:
    def __init__(
        self,
        *,
        chunk_size: int = 1200, 
        chunk_overlap: int = 200,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on = HEADERS_TO_SPLIT_ON,
            strip_headers = False,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators = TEXT_SEPARATOR
        )
    
    def prepare_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        return [
            chunk
            for document in documents
            for chunk in self._prepare_generic_document(document)
        ]

    def prepare_pdf_document(
        self,
        document: Document,
        source_blocks: Sequence[PdfSourceBlock],
    ) -> list[Document]:
        if not source_blocks:
            return self._prepare_generic_document(document)

        return self._prepare_pdf_document(document, source_blocks)

    def _prepare_generic_document(
        self,
        document: Document,
    ) -> list[Document]:
        paper_id = str(document.metadata["paper_id"])
        parent_id = str(document.metadata.get("parent_id", paper_id))

        markdown = self._clean_markdown(document.page_content)
        section_documents = self.header_splitter.split_text(markdown)

        for section in section_documents:
            section.metadata = {
                **document.metadata,
                **section.metadata,
                "paper_id": paper_id,
                "parent_id": parent_id,
                "section_path": self._build_section_path(section.metadata),
            }

        chunk_documents = self.text_splitter.split_documents(section_documents)

        return self._finalize_chunk_documents(chunk_documents, document)

    def _prepare_pdf_document(
        self,
        document: Document,
        source_blocks: Sequence[PdfSourceBlock],
    ) -> list[Document]:
        chunk_documents: list[Document] = []
        section_blocks: list[PdfSourceBlock] = []
        headings = ["", "", "", ""]
        section_path = ""

        for source_block in source_blocks:
            block = replace(
                source_block,
                markdown = self._clean_markdown(source_block.markdown),
            )
            if not block.markdown:
                continue

            heading = self._heading(block.markdown)
            if heading:
                if section_blocks:
                    chunk_documents.extend(
                        self._build_pdf_chunks(
                            document,
                            section_blocks,
                            section_path,
                        )
                    )
                    section_blocks = []

                level, title = heading
                headings[level - 1] = title
                for index in range(level, len(headings)):
                    headings[index] = ""
                section_path = " > ".join(
                    heading
                    for heading in headings
                    if heading
                )

            section_blocks.append(block)

        if section_blocks:
            chunk_documents.extend(
                self._build_pdf_chunks(
                    document,
                    section_blocks,
                    section_path,
                )
            )

        return self._finalize_chunk_documents(chunk_documents, document)

    def _build_pdf_chunks(
        self,
        document: Document,
        source_blocks: Sequence[PdfSourceBlock],
        section_path: str,
    ) -> list[Document]:
        chunks: list[Document] = []
        current: list[PdfSourceBlock] = []

        for source_block in source_blocks:
            for block in self._expand_pdf_block(source_block):
                if current and self._blocks_length(current + [block]) > self.chunk_size:
                    chunks.append(
                        self._build_pdf_chunk_document(
                            document,
                            current,
                            section_path,
                        )
                    )
                    current = self._overlap_tail(current)

                    if current and self._blocks_length(current + [block]) > self.chunk_size:
                        current = []

                current.append(block)

        if current:
            chunks.append(
                self._build_pdf_chunk_document(
                    document,
                    current,
                    section_path,
                )
            )

        return chunks

    def _expand_pdf_block(
        self,
        source_block: PdfSourceBlock,
    ) -> list[PdfSourceBlock]:
        if (
            len(source_block.markdown) <= self.chunk_size
            or source_block.block_type in PDF_ATOMIC_BLOCK_TYPES
        ):
            return [source_block]

        return [
            replace(source_block, markdown = fragment)
            for fragment in self.text_splitter.split_text(source_block.markdown)
            if fragment.strip()
        ]

    def _overlap_tail(
        self,
        source_blocks: Sequence[PdfSourceBlock],
    ) -> list[PdfSourceBlock]:
        if self.chunk_overlap == 0:
            return []

        overlap = []
        length = 0
        for block in reversed(source_blocks):
            overlap.append(block)
            length += len(block.markdown)
            if length >= self.chunk_overlap:
                break

        return list(reversed(overlap))

    def _build_pdf_chunk_document(
        self,
        document: Document,
        source_blocks: Sequence[PdfSourceBlock],
        section_path: str,
    ) -> Document:
        page_numbers = sorted({block.page_no for block in source_blocks})
        source_spans = []
        seen_source_indexes = set()

        for block in source_blocks:
            if block.source_index in seen_source_indexes:
                continue
            seen_source_indexes.add(block.source_index)

            span: dict[str, Any] = {
                "page": block.page_no,
                "type": block.block_type,
                "source_index": block.source_index,
            }
            if block.bbox is not None:
                span["bbox"] = block.bbox
            source_spans.append(span)

        return Document(
            page_content = "\n\n".join(
                block.markdown
                for block in source_blocks
            ),
            metadata = {
                **document.metadata,
                "section_path": section_path,
                "page_start": page_numbers[0],
                "page_end": page_numbers[-1],
                "page_numbers": page_numbers,
                "source_spans": source_spans,
            },
        )

    @staticmethod
    def _blocks_length(
        source_blocks: Sequence[PdfSourceBlock],
    ) -> int:
        return len("\n\n".join(block.markdown for block in source_blocks))

    @staticmethod
    def _heading(markdown: str) -> tuple[int, str] | None:
        match = re.match(r"^(#{1,4})\s+(.+)$", markdown)
        if not match:
            return None

        return len(match.group(1)), match.group(2).strip()

    def _finalize_chunk_documents(
        self,
        chunk_documents: list[Document],
        document: Document,
    ) -> list[Document]:
        paper_id = str(document.metadata["paper_id"])
        parent_id = str(document.metadata.get("parent_id", paper_id))

        for chunk_index, chunk in enumerate(chunk_documents):
            content = chunk.page_content.strip()
            chunk.page_content = content

            chunk.metadata.update(
                {
                    "paper_id": paper_id,
                    "parent_id": parent_id,
                    "chunk_id": self._build_chunk_id(
                        paper_id = paper_id,
                        chunk_index = chunk_index,
                        content = content,
                    ),
                    "chunk_index": chunk_index,
                    "doc_type": "child",
                }
            )

        return chunk_documents

    @staticmethod
    def _clean_markdown(
        markdown: str,
    ) -> str:
        cleaned = "\n".join(
            line.rstrip()
            for line in markdown.splitlines()
        )

        return re.sub(
            r"\n{3,}",
            "\n\n",
            cleaned,
        ).strip()

    @classmethod
    def _build_section_path(
        cls,
        metadata: dict[str, Any],
    ) -> str:
        return " > ".join(
            str(metadata[key])
            for key in HEADER_KEYS
            if metadata.get(key)
        )

    @staticmethod
    def _build_chunk_id(
        *,
        paper_id: str,
        chunk_index: int,
        content: str,
    ) -> str:
        value = (
            f"{paper_id}:"
            f"{chunk_index}:"
            f"{content}"
        )

        return hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()
