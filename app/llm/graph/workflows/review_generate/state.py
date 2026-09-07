from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class TaskReviewWorkflowState(TypedDict, total=False):
    run_id: str
    active_tool_call: dict[str, Any] | None
    messages: Annotated[list[BaseMessage], add_messages]
    last_action_result: dict[str, Any] | None
    artifact_refs: list[str]
    task_id: int
    topic: str
    language: str
    citation_style: str
    review_type: str
    allow_abstract_evidence: bool
    max_reflection_rounds: int
    reflection_round: int

    stage: str
    status: str
    error: str | None
    warnings: list[str]

    paper_ids_snapshot: list[str]
    corpus_artifact_ref: str | None

    framework_artifact_ref: str | None
    framework_hash: str | None

    claims_artifact_ref: str | None
    evidence_ledger_artifact_ref: str | None

    render_section_ids: list[str]
    retrieve_claim_ids: list[str]
    revise_claim_ids: list[str]

    section_draft_artifact_refs: dict[str, str]
    review_draft_artifact_ref: str | None

    reflection_report_artifact_ref: str | None
    revision_plan_artifact_ref: str | None

    final_review_artifact_ref: str | None
    review_id: int | None
    review_version_number: int
