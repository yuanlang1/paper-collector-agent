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
    ARXIV = "arXiv"
    DBLP = "DBLP"
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
    def validate_year_range(self) -> PromptUnderstandingArgs:
        if self.yearFrom is None and self.yearTo is None:
            return self
        if self.yearFrom is None or self.yearTo is None:
            raise ValueError("yearFrom 和 yearTo 必须同时为空或同时提供")
        if self.yearFrom > self.yearTo:
            raise ValueError("yearFrom 必须小于或等于 yearTo")
        return self


class ProviderQueryPlanArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_query: NonBlankString = Field(
        ...,
        max_length=1000,
        description="展示给用户的实际检索式。",
    )
    reasoning: NonBlankString = Field(
        ...,
        max_length=1000,
        description="简洁说明检索式如何由结构化意图推导得到。",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "对应来源的查询参数；不得包含 start、f、num、h、max_results、total_limit、max_pages；"
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
    query_plans: dict[
        Literal["arxiv", "dblp", "google_scholar"], ProviderQueryPlanArgs
    ] = Field(
        ...,
        description="每个可用 Provider 的查询草案，必须覆盖输入提供的来源。",
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
        description="允许的检索来源。",
    )

    @model_validator(mode="after")
    def remove_duplicate_tags(self) -> SearchTagArgs:
        self.paperTag = list(dict.fromkeys(self.paperTag))
        self.sourceTag = list(dict.fromkeys(self.sourceTag))
        return self


class PaperFrontPageMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(
        default=None,
        max_length=500,
        description=("当前论文的完整原文标题。允许修复 OCR 断行，不允许翻译或改写。"),
    )

    authors: list[str] = Field(
        default_factory=list,
        description=("按论文署名顺序排列的作者姓名列表，不包含机构、邮箱或上标符号。"),
    )

    doi: str | None = Field(
        default=None,
        description=(
            "当前论文的 DOI 标识符，不包含 https://doi.org/ 前缀。无法确认时返回 null。"
        ),
    )

    abstract: str | None = Field(
        default=None,
        max_length=10000,
        description=("论文明确标注的原文摘要。保留原文语言，不翻译、不扩写。"),
    )

    venue_name: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "论文明确标注的期刊或会议名称。不将出版社或作者机构误认为发表来源。"
        ),
    )

    keywords: list[str] = Field(
        default_factory=list,
        description=("论文明确标注的原文关键词，按原文顺序输出，不自行生成。"),
    )

    ai_abstract: str | None = Field(
        default=None,
        max_length=3000,
        description=(
            "基于论文前两页生成的中文 Markdown 学术摘要。"
            "必须按顺序包含四个三级标题："
            "### 研究背景与问题、"
            "### 研究目标、"
            "### 研究方法、"
            "### 主要贡献与发现。"
            "每节使用自然段，缺失信息明确标注。"
            "整体内容不足时返回 null。"
            "不得编造方法、实验结果或结论。"
        ),
    )


class PaperRecommendationResult(BaseModel):
    recommendation_stars: int = Field(
        ge=1,
        le=5,
        description="论文与用户检索意图的匹配星数，取 1 至 5 的整数，分数越高表示越匹配。",
    )
    recommendation_reason: str = Field(
        min_length=1,
        max_length=1_000,
        description="用中文说明具体匹配点或不匹配点，依据输入中的论文信息和检索意图。",
    )
