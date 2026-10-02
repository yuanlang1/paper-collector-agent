from __future__ import annotations

from enum import Enum, IntEnum
from typing import Annotated, Any, Literal

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
    GOOGLE_SCHOLAR = "Google Scholar"


PromptIntent = Literal["survey", "benchmark", "method", "mixed"]


class PromptUnderstandingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: NonBlankString = Field(
        ...,
        max_length=300,
        description="用户请求中提炼的核心研究主题，使用简洁的英文学术表达。",
    )
    subfields: list[NonBlankString] = Field(
        default_factory=list,
        description="用户明确提到的研究子方向，未提及时为空列表。",
    )
    intent: PromptIntent = Field(
        ...,
        description="检索意图：survey、benchmark、method 或 mixed。",
    )
    yearFrom: int | None = Field(
        default=None,
        ge=1900,
        le=2100,
        description="用户明确指定的起始年份，包含该年；必须与 yearTo 同时提供。",
    )
    yearTo: int | None = Field(
        default=None,
        ge=1900,
        le=2100,
        description="用户明确指定的结束年份，包含该年；必须与 yearFrom 同时提供。",
    )
    keywords: list[NonBlankString] = Field(
        default_factory=list,
        description="从用户请求提取的核心英文检索词。",
    )
    synonyms: list[NonBlankString] = Field(
        default_factory=list,
        description="keywords 的可靠同义词或常用英文学术表达。",
    )
    includeTerms: list[NonBlankString] = Field(
        default_factory=list,
        description="用户明确要求包含的英文术语，未提及时为空列表。",
    )
    excludeTerms: list[NonBlankString] = Field(
        default_factory=list,
        description="用户明确要求排除的英文术语，未提及时为空列表。",
    )
    requiresCode: bool | None = Field(
        default=None,
        description="用户明确要求代码、实现或开源仓库时为 true；未提及时为 null。",
    )
    reasoning: NonBlankString = Field(
        ...,
        max_length=1000,
        description="用中文简要说明各字段如何从用户请求得出，不得补充未给出的限制。",
    )

    @model_validator(mode="after")
    def validate_year_range(self) -> "PromptUnderstandingArgs":
        if self.yearFrom is None and self.yearTo is None:
            return self
        if self.yearFrom is None or self.yearTo is None:
            raise ValueError("yearFrom 和 yearTo 必须同时为空或同时提供")
        if self.yearFrom > self.yearTo:
            raise ValueError("yearFrom 必须小于或等于 yearTo")
        return self


class GoogleScholarQueryPlanArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_query: NonBlankString = Field(
        ...,
        max_length=1000,
        description="展示给用户的 Google Scholar 实际检索式。",
    )
    reasoning: NonBlankString = Field(
        ...,
        max_length=1000,
        description="简洁说明检索式如何由结构化意图推导得到。",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Google Scholar 查询参数，仅可包含 query、year_from、year_to、review_only；"
            "分页参数由代码注入。"
        ),
    )


class SearchPlanningArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_understanding: PromptUnderstandingArgs = Field(
        ...,
        description="从用户请求解析出的完整检索意图。",
    )
    paperTag: list[PaperTypeCode] = Field(
        default_factory=lambda: [
            PaperTypeCode.JOURNAL_ARTICLE,
            PaperTypeCode.PROCEEDINGS_ARTICLE,
        ],
        min_length=1,
        description="论文类型整数编码；未限定时使用 1（期刊论文）和 2（会议论文）。",
    )
    query_plan: GoogleScholarQueryPlanArgs = Field(
        ...,
        description="唯一的 Google Scholar 查询草案；来源由代码固定，不在模型输出中声明。",
    )


class SearchTagArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    yearTag: int = Field(
        default=0,
        ge=0,
        description="代码根据明确年份范围计算的检索年份标签；0 表示未限定年份。",
    )
    paperTag: list[PaperTypeCode] = Field(
        default_factory=lambda: [
            PaperTypeCode.JOURNAL_ARTICLE,
            PaperTypeCode.PROCEEDINGS_ARTICLE,
        ],
        min_length=1,
        description="允许的论文类型整数编码，默认包含期刊论文和会议论文。",
    )
    sourceTag: list[SourceTypeValue] = Field(
        default_factory=lambda: [SourceTypeValue.GOOGLE_SCHOLAR],
        min_length=1,
        description="允许的检索来源，当前仅支持 Google Scholar。",
    )

    @model_validator(mode="after")
    def remove_duplicate_tags(self) -> "SearchTagArgs":
        self.paperTag = list(dict.fromkeys(self.paperTag))
        self.sourceTag = list(dict.fromkeys(self.sourceTag))
        return self
