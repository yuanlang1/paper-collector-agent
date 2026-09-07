from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from app.llm.subagents.task_review.contracts import TaskReviewDelegation


def _initialization_failed(error: str) -> dict[str, Any]:
    return {
        "stage": "failed",
        "status": "failed",
        "error": error,
        "warnings": [],
    }


async def initialize_review_node(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    run_id = state.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        return _initialization_failed("missing valid run_id")

    call = state.get("active_tool_call")
    raw_request = call.get("args") if isinstance(call, Mapping) else None
    if not isinstance(raw_request, Mapping):
        return _initialization_failed("missing active task review delegation")

    try:
        request = TaskReviewDelegation.model_validate(raw_request)
    except ValidationError as exc:
        return _initialization_failed(f"invalid review request: {exc}")

    return {
        "task_id": request.task_id,
        "topic": request.topic,
        "language": request.language,
        "citation_style": request.citation_style,
        "review_type": request.review_type,
        "allow_abstract_evidence": False,
        "max_reflection_rounds": request.max_reflection_rounds,
        "stage": "loading_corpus",
        "status": "running",
        "error": None,
        "warnings": [],
        "reflection_round": 0,
        "paper_ids_snapshot": [],
        "corpus_artifact_ref": None,
        "framework_artifact_ref": None,
        "framework_hash": None,
        "claims_artifact_ref": None,
        "evidence_ledger_artifact_ref": None,
        "render_section_ids": [],
        "retrieve_claim_ids": [],
        "revise_claim_ids": [],
        "section_draft_artifact_refs": {},
        "review_draft_artifact_ref": None,
        "reflection_report_artifact_ref": None,
        "revision_plan_artifact_ref": None,
        "final_review_artifact_ref": None,
        "review_id": None,
        "review_version_number": 1,
    }
