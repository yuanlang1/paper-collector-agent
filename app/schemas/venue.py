from decimal import Decimal

from pydantic import BaseModel, Field


class VenueInfoDraft(BaseModel):
    standard_name: str = Field(min_length=1, max_length=500)
    acronym: str | None = Field(default=None, max_length=100)
    type: int = 0

    sci_rank: str | None = Field(default=None, max_length=50)
    ccf_rank: str | None = Field(default=None, max_length=50)
    sci_if: Decimal = Decimal(0)
    sci_up: str | None = Field(default=None, max_length=100)
    sci_up_small: str | None = Field(default=None, max_length=200)
    core_rank: str | None = Field(default=None, max_length=20)


class VenueRecord(VenueInfoDraft):
    id: int = Field(gt=0, strict=True)
