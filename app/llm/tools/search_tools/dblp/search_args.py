from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DblpPaperType = Literal[
    "all",
    "Journal Articles",
    "Conference and Workshop Papers",
    "Books and Theses",
    "Parts in Books or Collections",
    "Editorship",
    "Reference Works",
    "Data and Artifacts",
    "Informal and Other Publications",
]


class DblpSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "DBLP 出版物查询词，例如论文标题、作者名、研究主题或会议/期刊名称。"
        ),
    )

    f: int = Field(
        0,
        ge=0,
        description="DBLP 结果起始下标，从 0 开始。",
    )

    h: int = Field(
        5,
        ge=1,
        le=1000,
        description="DBLP 单次返回结果数；最大 1000。",
    )

    total_limit: int = Field(
        5,
        ge=1,
        description="最终最多返回的论文数量；结果不足时返回全部可检索结果。",
    )

    max_pages: int = Field(1, ge=1, le=20)

    year_from: int | None = Field(
        None,
        ge=1936,
        le=2100,
        description="可选起始年份；在 DBLP 返回结果上本地过滤。",
    )

    year_to: int | None = Field(
        None,
        ge=1936,
        le=2100,
        description="可选结束年份；在 DBLP 返回结果上本地过滤。",
    )

    venue: str | None = Field(
        None,
        min_length=1,
        max_length=200,
        description=(
            "可选会议或期刊名称过滤，例如 ACL、CVPR、NeurIPS、ICML；在返回结果上本地过滤。"
        ),
    )

    paper_type: DblpPaperType = Field(
        "all",
        description=(
            "可选出版物类型过滤；默认 all 表示不过滤。"
        ),
    )

    @model_validator(mode="after")
    def validate_year_range(self) -> "DblpSearchArgs":
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError(
                "year_from must be less than or equal to year_to"
            )

        return self
