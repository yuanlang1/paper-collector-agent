from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import PydanticCustomError


class IndexBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_ids: list[Annotated[int, Field(strict=True, gt=0)]] = Field(
        min_length=1, max_length=20,
    )

    @field_validator("paper_ids")
    @classmethod
    def unique_paper_ids(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise PydanticCustomError("duplicate_paper_ids", "paper_ids must be unique")
        return value
