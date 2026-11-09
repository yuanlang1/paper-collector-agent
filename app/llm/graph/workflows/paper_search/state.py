from __future__ import annotations

from typing import Annotated, Any, Literal

from typing_extensions import TypedDict

ProviderId = Literal["arxiv", "dblp", "google_scholar"]


def merge_search_results(left: dict, right: dict) -> dict:
    return {**left, **right}


class SearchTask(TypedDict):
    provider_id: ProviderId
    round: int
    query_args: dict[str, Any]
    configuration_version: str
    cursor: dict[str, Any] | None
    limit: int
    max_pages: int


class SearchWorkerState(TypedDict):
    search_task: SearchTask
    child_run_id: str


PaperSearchStage = Literal[
    "preparing_search",
    "searching",
    "normalizing",
    "reviewing_search",
    "enriching",
    "downloading_pdfs",
    "enriching_abstract",
    "uploading_oss",
    "persisting_papers",
    "resolving_venues",
    "recommending",
    "completed",
    "failed",
    "blocked",
]

PaperSearchStatus = Literal[
    "running",
    "completed",
    "failed",
    "blocked",
    "success",
    "timed_out",
]


class PaperSearchOutputState(TypedDict):
    content: str
    status: Literal["success", "failed", "timed_out"]


class PaperSearchWorkflowState(TypedDict, total=False):
    conversation_id: str
    run_id: str
    child_run_id: str
    prompt: str
    task_id: int | None
    content: str
    original_prompt: str

    provider_ids: list[ProviderId] | None
    provider_query_plans: dict[str, SearchTask]
    active_search_tasks: dict[str, SearchTask]
    search_results: Annotated[dict[str, dict[str, Any]], merge_search_results]
    search_history: dict[str, dict[str, dict[str, Any]]]
    provider_cursors: dict[str, dict[str, Any] | None]
    provider_stats: dict[str, dict[str, Any]]

    query_understanding: dict[str, Any] | None
    intent_error: str | None

    search_tag: dict[str, Any] | None

    query_plan_artifact_ref: str | None

    create_task_payload: dict[str, Any] | None

    local_task_id: str | None
    paper_service_task_id: int | None

    supplemental_search_round: int

    raw_result_artifact_refs: list[str]
    normalized_manifest_artifact_ref: str | None
    existing_papers_manifest_artifact_ref: str | None
    crossref_enrichment_manifest_artifact_ref: str | None
    pdf_manifest_artifact_ref: str | None
    failed_manifest_artifact_ref: str | None
    venue_manifest_artifact_ref: str | None
    abstract_manifest_artifact_ref: str | None
    recommendation_manifest_artifact_ref: str | None
    task_bound_recommendation_manifest_artifact_ref: str | None
    persisted_papers_manifest_artifact_ref: str | None
    downloaded_pdf_paths: list[str]
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
