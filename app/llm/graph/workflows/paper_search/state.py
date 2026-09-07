from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


SourceName = Literal["arXiv", "DBLP", "Crossref", "Google Scholar"]

PaperSearchStage = Literal[
    "intent_understanding",
    "building_search_tag",
    "generating_source_queries",
    "awaiting_confirmation",
    "searching",
    "normalizing",
    "reviewing_search",
    "planning_supplemental_search",
    "enriching",
    "enriching_crossref",
    "downloading_pdfs",
    "enriching_abstract",
    "uploading_oss",
    "persisting_papers",
    "resolving_venues",
    "recommending",
    "completed",
    "partial_failed",
    "failed",
    "blocked",
]

PaperSearchStatus = Literal[
    "running",
    "waiting_confirmation",
    "completed",
    "partial_failed",
    "failed",
    "blocked",
]


class SourceQueryPlan(TypedDict):
    source: SourceName
    display_query: str
    reasoning: str
    arguments: dict[str, Any]
    post_filters: dict[str, Any]


class SourceRunSummary(TypedDict):
    source: SourceName
    status: Literal["success", "partial", "failed"]
    discovered_count: int
    artifact_ref: str | None
    error: str | None


class PaperSearchWorkflowState(TypedDict, total=False):
    conversation_id: str
    run_id: str
    active_tool_call: dict[str, Any] | None
    messages: Annotated[list[BaseMessage], add_messages]
    last_action_result: dict[str, Any] | None
    artifact_refs: list[str]
    paper_search_source_limits: dict[str, int] | None
    original_prompt: str
    paper_search_constraints: dict[str, Any] | None

    requested_sources: list[SourceName]
    download_pdfs: bool

    query_understanding: dict[str, Any] | None
    intent_error: str | None

    search_tag: dict[str, Any] | None

    source_query_plans: list[SourceQueryPlan]
    query_plan_artifact_ref: str | None

    confirmation_required: bool
    confirmation_message: str | None
    confirmation_decision: Literal["approved", "rejected"] | None

    create_task_payload: dict[str, Any] | None

    local_task_id: str | None
    paper_service_task_id: int | None

    source_summaries: dict[str, SourceRunSummary]
    active_search_sources: list[SourceName]
    active_source_query_plans: list[SourceQueryPlan]
    source_search_cursors: dict[str, dict[str, Any]]
    source_search_stats: dict[str, dict[str, Any]]
    supplemental_search_round: int

    raw_result_artifact_refs: list[str]
    normalized_manifest_artifact_ref: str | None
    existing_papers_manifest_artifact_ref: str | None
    enrichment_manifest_artifact_ref: str | None
    crossref_enrichment_manifest_artifact_ref: str | None
    pdf_manifest_artifact_ref: str | None
    failed_manifest_artifact_ref: str | None
    venue_manifest_artifact_ref: str | None
    abstract_manifest_artifact_ref: str | None
    recommendation_manifest_artifact_ref: str | None
    task_bound_recommendation_manifest_artifact_ref: str | None
    persisted_papers_manifest_artifact_ref: str | None
    downloaded_pdf_paths: list[str]
    pdf_cleanup_error: str | None
    degraded: bool
    task_status_update_error: str | None
    remote_task_state: str | None
    search_quality_artifact_ref: str | None
    search_review_decision_artifact_ref: str | None
    search_review_decision: dict[str, Any] | None

    current_batch_index: int
    total_batches: int

    progress: dict[str, int]
    stage: PaperSearchStage
    status: PaperSearchStatus
    warnings: list[str]
    error: str | None
