
from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    conversation_id: str = Field(min_length = 1)
    message: str = Field(min_length = 1)