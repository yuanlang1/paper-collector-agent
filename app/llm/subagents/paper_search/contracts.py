from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


PaperSearchSource = Literal["arXiv", "DBLP", "Google Scholar"]


class PaperSearchConstraints(BaseModel):
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    year_to: int | None = Field(default=None, ge=1900, le=2100)
    sources: list[PaperSearchSource] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_year_range(self) -> "PaperSearchConstraints":
        if (self.year_from is None) != (self.year_to is None):
            raise ValueError("year_from and year_to must be provided together")
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from must not be after year_to")
        return self


class PaperSearchDelegation(BaseModel):
    prompt: str = Field(min_length=1, max_length=2_000)
    objective: str = "检索、筛选、推荐并保存相关论文"
    constraints: PaperSearchConstraints = Field(
        default_factory=PaperSearchConstraints
    )
