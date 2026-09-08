from __future__ import annotations

from enum import Enum, IntEnum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)


NonBlankString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class PaperTypeCode(IntEnum):
    JOURNAL_ARTICLE = 1
    PROCEEDINGS_ARTICLE = 2
    DISSERTATION = 3
    BOOK_SECTION = 4
    MONOGRAPH = 5
    REPORT_COMPONENT = 6
    REPORT = 7
    PEER_REVIEW = 8
    BOOK_TRACK = 9
    BOOK_PART = 10
    OTHER = 11
    BOOK = 12
    JOURNAL_VOLUME = 13
    BOOK_SET = 14
    REFERENCE_ENTRY = 15
    JOURNAL = 16
    COMPONENT = 17
    BOOK_CHAPTER = 18
    PROCEEDINGS_SERIES = 19
    REPORT_SERIES = 20
    PROCEEDINGS = 21
    DATABASE = 22
    STANDARD = 23
    REFERENCE_BOOK = 24
    POSTED_CONTENT = 25
    JOURNAL_ISSUE = 26
    GRANT = 27
    DATASET = 28
    BOOK_SERIES = 29
    EDITED_BOOK = 30

    @property
    def api_value(self) -> str:
        return self.name

    @property
    def crossref_type(self) -> str:
        return self.name.lower().replace("_", "-")

    def to_option(self) -> dict[str, int | str]:
        return {
            "code": int(self),
            "value": self.api_value,
            "crossrefType": self.crossref_type,
        }

    @classmethod
    def _missing_(cls, value: object):
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        if normalized.isdecimal():
            return cls(int(normalized))
        member_name = normalized.upper().replace("-", "_").replace(" ", "_")
        return cls.__members__.get(member_name)


class SourceTypeValue(str, Enum):
    ARXIV = "arXiv"
    DBLP = "DBLP"
    CROSSREF = "Crossref"
    GOOGLE_SCHOLAR = "Google Scholar"


PromptIntent = Literal["survey", "benchmark", "method", "mixed"]


class PromptUnderstandingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: NonBlankString = Field(..., max_length=300)
    subfields: list[NonBlankString] = Field(default_factory=list)
    intent: PromptIntent
    yearFrom: int | None = Field(default=None, ge=1900, le=2100)
    yearTo: int | None = Field(default=None, ge=1900, le=2100)
    keywords: list[NonBlankString] = Field(default_factory=list)
    synonyms: list[NonBlankString] = Field(default_factory=list)
    includeTerms: list[NonBlankString] = Field(default_factory=list)
    excludeTerms: list[NonBlankString] = Field(default_factory=list)
    requiresCode: bool | None = None
    reasoning: NonBlankString = Field(..., max_length=1000)

    @model_validator(mode="after")
    def validate_year_range(self) -> "PromptUnderstandingArgs":
        if self.yearFrom is None and self.yearTo is None:
            return self
        if self.yearFrom is None or self.yearTo is None:
            raise ValueError("yearFrom 和 yearTo 必须同时为空或同时提供")
        if self.yearFrom > self.yearTo:
            raise ValueError("yearFrom 必须小于或等于 yearTo")
        return self


class SearchTagArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    yearTag: int = Field(default=0, ge=0)
    paperTag: list[PaperTypeCode] = Field(
        default_factory=lambda: [
            PaperTypeCode.JOURNAL_ARTICLE,
            PaperTypeCode.PROCEEDINGS_ARTICLE,
        ],
        min_length=1,
    )
    sourceTag: list[SourceTypeValue] = Field(
        default_factory=lambda: [
            SourceTypeValue.ARXIV,
            SourceTypeValue.DBLP,
            SourceTypeValue.GOOGLE_SCHOLAR,
        ],
        min_length=1,
    )

    @model_validator(mode="after")
    def remove_duplicate_tags(self) -> "SearchTagArgs":
        self.paperTag = list(dict.fromkeys(self.paperTag))
        self.sourceTag = list(dict.fromkeys(self.sourceTag))
        return self
