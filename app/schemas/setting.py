from pydantic import BaseModel, ConfigDict, Field


class PostAddModelRequest(BaseModel):
    user_id: str
    provider: str
    model_name: str
    base_url: str
    api_key: str
    is_small_model: bool
    is_default: bool = False


class ModelSettingListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    provider: str
    model_name: str = Field(validation_alias="model")
    base_url: str
    is_small_model: bool
    is_default: bool
