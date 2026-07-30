from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ArxivSupplementArgs(BaseModel):
    """
    arXiv 论文字段补全工具。

    用于补全 DBLP / Google Scholar 检索结果中缺失的字段：
    - summary
    - pdf_url
    - abstract_url
    - arxiv_id
    """

    papers: list[dict[str, Any]] = Field(
        ...,
        description="待补全论文列表，通常来自 dblp_search 或 google_scholar_search 的 papers 字段。",
    )

    fill_fields: list[Literal[
        "summary",
        "pdf_url",
        "abstract_url",
        "arxiv_id",
        "published",
        "updated",
        "authors",
        "categories",
        "doi",
    ]] = Field(
        default_factory=lambda: [
            "summary",
            "pdf_url",
            "abstract_url",
            "arxiv_id",
        ],
        description="需要从 arXiv 补全的字段。",
    )

    match_threshold: float = Field(
        0.88,
        description="标题模糊匹配阈值，0~1，越高越严格。",
    )

    max_candidates: int = Field(
        3,
        description="每篇论文最多从 arXiv 拉取多少个候选结果。",
    )

    only_when_missing: bool = Field(
        True,
        description="是否只在原字段为空时补充。建议保持 True，避免覆盖 DBLP / Google Scholar 原始结果。",
    )