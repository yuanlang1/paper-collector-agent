from enum import Enum
from typing import Annotated, Literal
from enum import IntEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)


NonBlankString = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
    ),
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

        member_name = (
            normalized
            .upper()
            .replace("-", "_")
            .replace(" ", "_")
        )
        return cls.__members__.get(member_name)


class SourceTypeValue(str, Enum):
    ARXIV = "arXiv"
    DBLP = "DBLP"
    CROSSREF = "Crossref"
    GOOGLE_SCHOLAR = "Google Scholar"


PromptIntent = Literal[
    "survey",
    "benchmark",
    "method",
    "mixed",
]


class PromptUnderstandingArgs(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    topic: NonBlankString = Field(
        ...,
        max_length=300,
        description="论文检索主题，不能为空，长度不能超过 300。",
    )

    subfields: list[NonBlankString] = Field(
        default_factory=list,
        description="研究主题所属的子领域列表。",
    )

    intent: PromptIntent = Field(
        ...,
        description=(
            "用户检索意图。"
            "survey=综述，benchmark=基准或数据集，"
            "method=方法研究，mixed=混合意图。"
        ),
    )

    yearFrom: int | None = Field(
        default=None,
        ge=1900,
        le=2100,
        description=(
            "检索起始年份。yearFrom 和 yearTo 必须同时为空或者同时提供。"
        ),
    )

    yearTo: int | None = Field(
        default=None,
        ge=1900,
        le=2100,
        description=(
            "检索结束年份。yearFrom 和 yearTo 必须同时为空或者同时提供。"
        ),
    )

    keywords: list[NonBlankString] = Field(
        default_factory=list,
        description="核心检索关键词。",
    )

    synonyms: list[NonBlankString] = Field(
        default_factory=list,
        description="核心关键词的同义词或近义表达。",
    )

    includeTerms: list[NonBlankString] = Field(
        default_factory=list,
        description="检索结果必须包含的术语。",
    )

    excludeTerms: list[NonBlankString] = Field(
        default_factory=list,
        description="检索结果需要排除的术语。",
    )

    requiresCode: bool | None = Field(
        default=None,
        description=(
            "是否要求论文提供代码。null 表示用户未明确提出代码要求。"
        ),
    )

    reasoning: NonBlankString = Field(
        ...,
        max_length=1000,
        description=("对用户检索要求进行结构化解析的理由，不能为空，长度不能超过 1000。"),
    )

    @model_validator(mode="after")
    def validate_year_range(self) -> "PromptUnderstandingArgs":
        year_from = self.yearFrom
        year_to = self.yearTo

        if year_from is None and year_to is None:
            return self

        if year_from is None or year_to is None:
            raise ValueError("yearFrom 和 yearTo 必须同时为空或同时提供")

        if year_from > year_to:
            raise ValueError("yearFrom 必须小于或等于 yearTo")

        return self


class SearchTagArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    yearTag: int = Field(
        default=0,
        ge=0,
        description="检索年限范围，0 代表没有年限范围，3 即为最近三年。",
    )

    paperTag: list[PaperTypeCode] = Field(
        default_factory=lambda: [
            PaperTypeCode.JOURNAL_ARTICLE,
            PaperTypeCode.PROCEEDINGS_ARTICLE,
        ],
        min_length=1,
        description="论文类型 code，和 paper-service 的 PaperType.code 对应。",
    )

    sourceTag: list[SourceTypeValue] = Field(
        default_factory=lambda: [
            SourceTypeValue.ARXIV,
            SourceTypeValue.DBLP,
            SourceTypeValue.GOOGLE_SCHOLAR,
        ],
        min_length=1,
        description="检索来源；默认检索 arXiv、DBLP 和 Google Scholar。",
    )

    @model_validator(mode="after")
    def remove_duplicate_tags(self) -> "SearchTagArgs":
        self.paperTag = list(dict.fromkeys(self.paperTag))
        self.sourceTag = list(dict.fromkeys(self.sourceTag))
        return self


class AddQueryTaskArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: NonBlankString = Field(..., description="用户原始的论文检索要求。")

    searchTag: SearchTagArgs = Field(..., description="论文年份、类型和检索来源标签。")

    queryUnderstanding: PromptUnderstandingArgs = Field(
        ...,
        description=("对用户检索要求的结构化理解，对应 paper-service 的 PromptUnderstandingDTO。")
    )


class GetTaskRagStatusArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: int = Field(..., gt=0, description="要查询 RAG 状态的检索任务 ID。")
