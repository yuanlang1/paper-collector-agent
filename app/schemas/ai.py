from datetime import datetime
from typing import Any

from openai.types.shared import ReasoningEffort
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from app.db.models.agent_run import RunStatus


class RunMetadata(BaseModel):
    model_id: int = Field(gt=0)
    think_effort: ReasoningEffort = None


class RunInput(BaseModel):
    message: str = Field(min_length=1)


class ChatRequest(BaseModel):
    user_id: int = Field(gt=0)
    metadata: RunMetadata
    thread_id: str | None = Field(default=None, max_length=128)
    input: RunInput | None = None
    resume: Any = None

    @model_validator(mode="after")
    def validate_input(self):
        if (self.input is not None) == ("resume" in self.model_fields_set):
            raise PydanticCustomError("run_input", "input 和 resume 必须且只能提供一个")
        if self.thread_id == "":
            self.thread_id = None
        if "resume" in self.model_fields_set and self.thread_id is None:
            raise PydanticCustomError("run_thread", "恢复请求必须提供 thread_id")
        return self


class CreateThreadRequest(BaseModel):
    user_id: int = Field(gt=0)
    topic: str | None = Field(default=None, max_length=500)


class RunInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: str
    thread_id: str
    user_id: int
    status: RunStatus
    metadata: dict[str, Any] = Field(validation_alias="run_metadata")
    started_at: datetime | None
    finished_at: datetime | None
    error_message: str | None
    creation_time: datetime
    modification_time: datetime
