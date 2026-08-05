from typing import TypedDict


class TaskReviewRequestState(TypedDict):
    task_id: int
    topic: str
    language: str
    citation_style: str
    review_type: str
    allow_abstract_evidence: bool
    max_reflection_rounds: int


class TaskReviewWorkflowState(TypedDict):
    run_id: str
    task_id: int
    topic: str
    language: str
    citation_style: str
    review_type: str
    allow_abstract_evidence: bool
    max_reflection_rounds: int
    reflection_round: int

    # 阶段与结果
    stage: str
    status: str
    error: str | None
    warnings: list[str]

    # 固定语料范围
    paper_ids_snapshot: list[str]
    corpus_artifact_ref: str | None

    # Framework
    framework_artifact_ref: str | None
    framework_hash: str | None

    # Claims 与 Evidence
    claims_artifact_ref: str | None
    evidence_ledger_artifact_ref: str | None

    # 定向修订，不重写无关章节
    render_section_ids: list[str]
    retrieve_claim_ids: list[str]
    revise_claim_ids: list[str]

    # 章节与最终正文
    section_draft_artifact_refs: dict[str, str]
    review_draft_artifact_ref: str | None

    # 反思与修订
    reflection_report_artifact_ref: str | None
    revision_plan_artifact_ref: str | None

    # 最终交接
    final_review_artifact_ref: str | None
    handoff: dict | None


