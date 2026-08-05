from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, ValidationError, field_validator


CitationStyle = Literal[
    "harvard",
    "apa",
    "ieee",
    "chicago",
    "vancouver"
]

ReviewType = Literal[
    "narrative",
    "systematic",
    "scoping",
    "critical",
]

class ReviewInitializeInput(BaseModel):
    task_id: int = Field(..., gt = 0, strict = True)
    topic: str = Field(..., min_length = 2, max_length = 2_000)

    language: Literal["zh-CN", "en"] = "zh-CN"
    citation_style: CitationStyle = "harvard"
    review_type: ReviewType = "narrative"

    allow_abstract_evidence: bool = False
    max_reflection_rounds: int = Field(
        default = 2,
        ge = 1,
        le = 3,
    )

    @field_validator("topic")
    @classmethod
    def normalize_topic(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("topic must not be blank")
        return value


def _initialization_failed(error: str) -> dict:
    return {
        "stage": "failed",
        "status": "failed",
        "error": error,
        "warnings": [],
        "handoff": None,
    }


async def initialize_review_node(
    state: dict
) -> dict:
    run_id = state.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        return _initialization_failed("missing valid run_id")

    raw_request = {
        "task_id": state.get("task_id"),
        "topic": state.get("topic"),
        "language": state.get("language", "zh-CN"),
        "citation_style": state.get("citation_style", "harvard"),
        "review_type": state.get("review_type", "narrative"),
        "allow_abstract_evidence": state.get(
            "allow_abstract_evidence",
            False,
        ),
        "max_reflection_rounds": state.get(
            "max_reflection_rounds",
            2,
        ),
    }

    try:
        request = ReviewInitializeInput.model_validate(raw_request)
    except ValidationError as exc:
        return _initialization_failed(f"invalid review request: {exc}")

    return {
        "task_id": request.task_id,
        "topic": request.topic,
        "language": request.language,
        "citation_style": request.citation_style,
        "review_type": request.review_type,
        "allow_abstract_evidence": request.allow_abstract_evidence,
        "max_reflection_rounds": request.max_reflection_rounds,

        "stage": "loading_corpus",
        "status": "running",
        "error": None,
        "warnings": [],
        "reflection_round": 0,

        # 论文集
        "paper_ids_snapshot": [],
        "corpus_artifact_ref": None,

        # Framework / Claims / Evidence
        "framework_artifact_ref": None,
        "framework_hash": None,
        "claims_artifact_ref": None,
        "evidence_ledger_artifact_ref": None,

        # 初次写作默认处理全部章节；反思后才改为定向修订
        "render_section_ids": [],
        "retrieve_claim_ids": [],
        "revise_claim_ids": [],

        # 草稿与质量报告
        "section_draft_artifact_refs": {},
        "review_draft_artifact_ref": None,
        "reflection_report_artifact_ref": None,
        "revision_plan_artifact_ref": None,

        # 最终结果
        "final_review_artifact_ref": None,
        "handoff": None,
    }
