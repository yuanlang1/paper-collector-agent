from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from app.llm.graph.workflows.review_generate.nodes.finalize import failed
from app.llm.subagents.task_review import TaskReviewDelegation


async def initialize_review_node(state: Mapping[str, Any],) -> dict[str, Any]:
    run_id = state.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        return failed("missing valid run_id", warnings=[])

    call = state.get("active_tool_call")
    raw_request = call.get("args") if isinstance(call, Mapping) else None
    if not isinstance(raw_request, Mapping):
        return failed("missing active task review delegation", warnings=[])

    try:
        request = TaskReviewDelegation.model_validate(raw_request)
    except ValidationError as exc:
        return failed(f"invalid review request: {exc}", warnings=[])

    return {
        "task_id": request.task_id,
        "topic": request.topic,
        "language": request.language,
        "citation_style": request.citation_style,
        "review_type": request.review_type,
        "allow_abstract_evidence": False,
        "max_reflection_rounds": request.max_reflection_rounds,
        "max_writing_revision_rounds": request.max_writing_revision_rounds,
        "stage": "loading_corpus",
        "status": "running",
        "error": None,
        "warnings": [],
        "error_code": None,
        "retryable": False,
        "claim_verification_artifact_ref": None,
        "revision_items": [],
        "revision_before": None,
        "reflection_round": 0,
        "writing_revision_round": 0,
        "paper_ids_snapshot": [],
        "corpus_artifact_ref": None,
        "study_records_artifact_ref": None,
        "study_extraction_report_artifact_ref": None,
        "reading_summary": {"total": 0, "succeeded": 0, "failed": 0},
        "framework_artifact_ref": None,
        "framework_hash": None,
        "changed_section_ids": [],
        "claims_artifact_ref": None,
        "evidence_ledger_artifact_ref": None,
        "section_draft_artifact_refs": {},
        "review_draft_artifact_ref": None,
        "reflection_report_artifact_ref": None,
        "writing_review_plan_ref": None,
        "revision_plan_artifact_ref": None,
        "final_review_artifact_ref": None,
        "review_id": None,
        "review_version_number": 1,
    }
