
from typing import Any, Literal
from pydantic import BaseModel


class EventBase(BaseModel):
    session_id: str
    run_id: str
    ts: str

class RunStartedEvent(EventBase):
    type: Literal["run.start"] = "run.start"

class LlmTokenEvent(EventBase):
    type: Literal["llm.token"] = "llm.token"
    text: str

class RunFinishedEvent(EventBase):
    type: Literal["run.finished"] = "run.finished"
    status: str = ["running", "failed"]
    reason: str

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


class SubagentStartedEvent(EventBase):
    type: Literal["subagent_started"] = "subagent_started"
    parent_run_id: str
    parent_session_id: str
    name: str
    description: str


class SubagentProgressEvent(EventBase):
    type: Literal["subagent_progress"] = "subagent_progress"
    parent_run_id: str
    parent_session_id: str
    name: str
    description: str


class SubagentFinishedEvent(EventBase):
    type: Literal["subagent_finished"] = "subagent_finished"
    parent_run_id: str
    parent_session_id: str
    name: str
    description: str
    status: Literal["success", "failed", "timed_out", "cancelled"]

    
