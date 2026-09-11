from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel, Field, HttpUrl


class FileType(str, Enum):
    PDF = "pdf"
    DOC = "doc"
    DOCX = "docx"
    PPT = "ppt"
    PPTX = "pptx"
    XLS = "xls"
    XLSX = "xlsx"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    JP2 = "jp2"
    WEBP = "webp"
    GIF = "gif"
    BMP = "bmp"
    HTML = "html"


class FileParseOptions(BaseModel):
    language: str = "en"

    is_ocr: bool = True
    enable_formula: bool = True
    enable_table: bool = True

    page_ranges: str | None = None
    extra_formats: list[str] = Field(default_factory=list)

    no_cache: bool = False
    cache_tolerance: int = 900


class FileParseRequest(BaseModel):
    file_id: str
    uri: HttpUrl

    file_name: str | None = None
    file_type: FileType | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
    options: FileParseOptions = Field(default_factory=FileParseOptions)


@dataclass(frozen=True)
class BatchDocumentParseResult:
    total: int
    documents: list["ParsedFileDocument"]
    errors: dict[str, str]

    @property
    def succeeded(self) -> int:
        return len(self.documents)

    @property
    def failed(self) -> int:
        return len(self.errors)


@dataclass(frozen=True, slots=True)
class PdfSourceBlock:
    markdown: str
    page_no: int
    bbox: list[float] | None
    block_type: str
    source_index: int


@dataclass(frozen=True, slots=True)
class ParsedFileDocument:
    document: Document
    pdf_source_blocks: tuple[PdfSourceBlock, ...] = ()
