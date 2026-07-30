from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ArxivSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(
        None,
        min_length=1,
        max_length=500,
        description="主题关键词、论文标题、作者名、摘要关键词或分类。",
    )

    arxiv_ids: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="可选 arXiv ID 列表，例如 ['2401.12345', 'hep-th/9901001']。提供后优先使用 arXiv API 的 id_list 参数精确查询。"
    )

    search_type: Literal[
        "topic",
        "title",
        "author",
        "abstract",
        "category",
    ] = Field(
        "topic",
        description="topic=主题；title=标题；author=作者；abstract=摘要；category=arXiv 分类。",
    )

    category: str | None = Field(
        None,
        max_length=200,
        description="可选分类过滤，如 cs.AI 或 cs.AI,cs.CL。",
    )

    year_from: int | None = Field(None, ge=1991, le=2100)
    year_to: int | None = Field(None, ge=1991, le=2100)

    start: int = Field(
        0,
        ge=0,
        description="arXiv 结果起始下标，从 0 开始。",
    )

    max_results: int = Field(
        5,
        ge=1,
        le=2000,
        description="arXiv 单次返回结果数；最大 2000。",
    )

    total_limit: int = Field(
        5,
        ge=1,
        le=30_000,
        description="最终最多返回的论文数量；结果不足时返回全部可检索结果。",
    )

    max_pages: int = Field(1, ge=1, le=20)

    sort: Literal["relevance", "newest", "updated"] = Field(
        "relevance",
        description="relevance=相关性；newest=按首次提交时间；updated=按最近更新日期。", 
    )

    include_abstract: bool = Field(True, description="是否返回摘要。")

    @model_validator(mode="after")
    def validate_request(self) -> "ArxivSearchArgs":
        if not self.query and not self.arxiv_ids:
            raise ValueError("query or arxiv_ids must be provided")

        if self.query and self.arxiv_ids:
            raise ValueError(
                "query and arxiv_ids cannot be provided together"
            )

        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError(
                "year_from must be less than or equal to year_to"
            )

        return self
