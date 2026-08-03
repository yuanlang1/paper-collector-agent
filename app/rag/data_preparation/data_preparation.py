
import hashlib
import logging
import re
from typing import Any
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

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

class DataPreparationModule:
    def __init__(
        self,
        *,
        chunk_size: int = 1200, 
        chunk_overlap: int = 200,
    ) -> None:
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
            for chunk in self._prepare_document(document)
        ]
    
    # prepare document
    def _prepare_document(
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