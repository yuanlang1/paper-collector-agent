from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.llm.graph.main.schemas import RunStatus
from app.memory.schemas import MemoryUsage
class PendingToolCallState(TypedDict):
    id: str
    name: str
    args: dict[str, Any]
    kind: Literal["tool", "subagent"] | None
    requires_confirmation: bool
    batch_id: str
    batch_calls: list["PendingToolCallState"]


class MainAgentState(TypedDict):
    conversation_id: str
    user_id: str
    run_id: str
    llm_profile: dict[str, Any] | None
    memory_llm_profile: dict[str, Any] | None

    paper_search_source_limits: dict[str, int] | None

    messages: Annotated[list[BaseMessage], add_messages]
    conversation_window_start_id: str | None
    system_context: str
    memory_usage: MemoryUsage | None

    pending_tool_calls: list[PendingToolCallState]
    active_tool_call: PendingToolCallState | None
    tool_batch_id: str | None
    tool_batch_calls: list[PendingToolCallState]
    iteration_count: int
    reasoning_content: str

    last_action_result: dict[str, Any] | None
    artifact_refs: list[str]

    reply: str
    run_status: RunStatus
    error: str | None
