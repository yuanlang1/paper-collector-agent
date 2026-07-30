from pydantic import BaseModel, ConfigDict, Field, model_validator


class GoogleScholarSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Google Scholar 检索词，可使用普通关键词、论文标题，也可使用 Google Scholar 支持的 author: 或 source: 辅助语法。"
        ),
    )

    start: int = Field(
        0,
        ge=0,
        description="SerpApi Google Scholar 结果偏移量，从 0 开始。",
    )

    num: int = Field(
        5,
        ge=1,
        le=20,
        description="SerpApi Google Scholar 单次返回结果数；最大 20。",
    )

    total_limit: int = Field(
        5,
        ge=1,
        description="最终最多返回的论文数量；结果不足时返回全部可检索结果。",
    )

    max_pages: int = Field(1, ge=1, le=20)

    year_from: int | None = Field(
        None,
        ge=1900,
        le=2100,
        description="可选起始年份，映射 SerpApi 的 as_ylo。",
    )

    year_to: int | None = Field(
        None,
        ge=1900,
        le=2100,
        description="可选结束年份，映射 SerpApi 的 as_yhi。",
    )

    review_only: bool = Field(
        False,
        description=(
            "是否仅检索综述类文章，映射 SerpApi 的 as_rr=1。"
        ),
    )

    @model_validator(mode="after")
    def validate_year_range(
        self,
    ) -> "GoogleScholarSearchArgs":
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError(
                "year_from must be less than or equal to year_to"
            )

        return self
