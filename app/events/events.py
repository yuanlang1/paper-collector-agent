
from typing import Any, Literal
from pydantic import BaseModel


class EventBase(BaseModel):
    session_id: str
    run_id: str
    ts: str

class ToolCallStartedEvent(EventBase):
    type: Literal["tool.call_started"] = "tool.call_started"
    tool_use_id: str
    tool_name: str
    params: dict[str, Any]

class ToolCallFailedEvent(EventBase):
    type: Literal["tool.call_failed"] = "tool.call_failed"
    tool_use_id: str
    tool_name: str
    error_type: str
    error_message: str
    elapsed_ms: int
    attempt: int = 1

class ToolCallFinishedEvent(EventBase):
    type: Literal["tool.call_finished"] = "tool.call_finished"
    tool_use_id: str
    tool_name: str
    elapsed_ms: int