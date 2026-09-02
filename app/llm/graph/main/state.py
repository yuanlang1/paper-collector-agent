from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.llm.graph.main.schemas import RunStatus
from app.llm.subagents.paper_search.contracts import (
    PaperSearchHandoffState,
    PaperSearchRequestState,
)
from app.llm.subagents.task_review.contracts import (
    TaskReviewHandoffState,
    TaskReviewRequestState,
)


class PendingToolCallState(TypedDict):
    id: str
    name: str
    args: dict[str, Any]
    kind: Literal["tool", "subagent"] | None
    requires_confirmation: bool


class MainAgentState(TypedDict):
    conversation_id: str
    run_id: str
    llm_profile: dict[str, Any] | None
    paper_search_request: PaperSearchRequestState | None
    paper_search_handoff: PaperSearchHandoffState | None
    paper_search_tool_call_id: str | None
    task_review_request: TaskReviewRequestState | None
    task_review_handoff: TaskReviewHandoffState | None
    task_review_tool_call_id: str | None

    messages: Annotated[list[BaseMessage], add_messages]

    pending_tool_calls: list[PendingToolCallState]
    active_tool_call: PendingToolCallState | None
    iteration_count: int
    reasoning_content: str

    last_action_result: dict[str, Any] | None
    artifact_refs: list[str]

    reply: str
    run_status: RunStatus
    error: str | None
