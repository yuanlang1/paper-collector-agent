from pydantic import BaseModel


class PostAddModelRequest(BaseModel):
    user_id: str
    provider: str
    model_name: str
    base_url: str
    api_key: str
    is_small_model: bool
