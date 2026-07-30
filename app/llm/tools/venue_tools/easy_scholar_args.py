from pydantic import BaseModel, ConfigDict, Field


class EasyScholarVenueArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    publication_name: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="要查询的期刊名称，例如 Nature、IEEE Transactions on Pattern Analysis and Machine Intelligence。",
    )