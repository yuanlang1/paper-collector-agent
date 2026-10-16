from typing import Any, Literal

from typing_extensions import TypedDict


class TaskReviewOutputState(TypedDict):
    content: str
    status: Literal["success", "failed", "timed_out"]


class TaskReviewWorkflowState(TypedDict, total=False):
    run_id: str
    child_run_id: str
    content: str
    task_id: int
    topic: str
    language: str
    citation_style: str
    review_type: str
    allow_abstract_evidence: bool
    reflection_round: int
    writing_revision_round: int

    stage: str
    status: str
    error: str | None
    warnings: list[str]

    paper_ids_snapshot: list[str]
    corpus_artifact_ref: str | None
    study_records_artifact_ref: str | None
    study_extraction_report_artifact_ref: str | None
    reading_summary: dict[str, int]

    framework_artifact_ref: str | None

    claims_artifact_ref: str | None
    evidence_ledger_artifact_ref: str | None

    section_draft_artifact_refs: dict[str, str]
    review_draft_artifact_ref: str | None

    reflection_report_artifact_ref: str | None
    writing_review_plan_ref: str | None
    revision_plan_artifact_ref: str | None

    final_review_artifact_ref: str | None
    review_id: int | None
    review_version_number: int

    error_code: str | None
    retryable: bool
    claim_verification_artifact_ref: str | None
    revision_items: list[dict[str, Any]]
    revision_before: str | None
