from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import PydanticCustomError


class IndexBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_id: str = Field(min_length=1)
    paper_ids: list[Annotated[int, Field(strict=True, gt=0)]] = Field(min_length=1)

    @field_validator("paper_ids")
    @classmethod
    def unique_paper_ids(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise PydanticCustomError("duplicate_paper_ids", "paper_ids must be unique")
        return value


class PostPaperProfile(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    user_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    oss_name: str = Field(min_length=1)
