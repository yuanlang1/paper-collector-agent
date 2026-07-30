from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.llm.graph.main.schemas import (
    ActionExecutionRecordState,
    ActionResultState,
    ConfirmationRecordState,
    PendingActionState,
    RunStatus,
    SolveDecisionState,
)
from app.llm.subagents.paper_search.contracts import (
    PaperSearchHandoffState,
    PaperSearchRequestState,
)


class MainAgentState(TypedDict):
    conversation_id: str
    run_id: str
    forced_subagent: dict[str, Any] | None
    paper_search_request: PaperSearchRequestState | None
    paper_search_handoff: PaperSearchHandoffState | None
    paper_search_action_id: str | None

    messages: Annotated[list[BaseMessage], add_messages]

    solve_decision: SolveDecisionState | None
    pending_action: PendingActionState | None
    confirmation: ConfirmationRecordState | None

    last_action_result: ActionResultState | None
    action_history: list[ActionExecutionRecordState]
    artifact_refs: list[str]

    reply: str
    run_status: RunStatus
    error: str | None

    action_rounds: int
    max_action_rounds: int
