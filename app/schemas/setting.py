from pydantic import BaseModel, ConfigDict, Field


class PostAddModelRequest(BaseModel):
    user_id: str
    provider: str
    model_name: str
    base_url: str
    api_key: str
    is_small_model: bool


class UpdateSourceLimitsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arxiv: int = Field(strict=True, ge=1)
    dblp: int = Field(strict=True, ge=1)
    google_scholar: int = Field(strict=True, ge=1)
