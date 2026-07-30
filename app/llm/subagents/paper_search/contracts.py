from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator


PaperSearchSource = Literal["arXiv", "DBLP", "Google Scholar"]


class PaperSearchConstraints(BaseModel):
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    year_to: int | None = Field(default=None, ge=1900, le=2100)
    sources: list[PaperSearchSource] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_year_range(self) -> "PaperSearchConstraints":
        if (self.year_from is None) != (self.year_to is None):
            raise ValueError("year_from and year_to must be provided together")
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from must not be after year_to")
        return self


class PaperSearchDelegation(BaseModel):
    prompt: str = Field(min_length=1, max_length=2_000)
    objective: str = "检索、筛选、推荐并保存相关论文"
    constraints: PaperSearchConstraints = Field(
        default_factory=PaperSearchConstraints
    )


class PaperSearchHandoff(BaseModel):
    status: Literal["success", "partial", "error", "rejected"]
    summary: str
    data: dict[str, Any]
    artifact_refs: list[str]
    retryable: bool
    error_code: str | None = None
    error_message: str | None = None


class PaperSearchRequestState(TypedDict):
    prompt: str
    objective: str
    constraints: dict[str, Any]


class PaperSearchHandoffState(TypedDict):
    status: Literal["success", "partial", "error", "rejected"]
    summary: str
    data: dict[str, Any]
    artifact_refs: list[str]
    retryable: bool
    error_code: str | None
    error_message: str | None


def _artifact_refs(state: Mapping[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            value
            for key, value in state.items()
            if key.endswith("_artifact_ref")
            and isinstance(value, str)
            and value.startswith("artifact://")
        )
    )


def build_paper_search_handoff(
    state: Mapping[str, Any],
) -> PaperSearchHandoff:
    stage = str(state.get("stage"))
    progress = state.get("progress") or {}
    status = {
        "completed": "success",
        "partial_failed": "partial",
    }.get(stage, "error")
    if (
        stage == "blocked"
        and state.get("confirmation_decision") == "rejected"
    ):
        status = "rejected"
    if stage == "completed":
        summary = (
            f"论文检索完成：推荐 {progress.get('recommendation_kept', 0)} 篇，"
            f"保存关系 {progress.get('persistence_saved', 0)} 条。"
        )
    elif stage == "partial_failed":
        summary = "论文检索部分完成，请查看警告与结果 artifact。"
    else:
        summary = "论文检索未完成。"
    return PaperSearchHandoff(
        status=status,
        summary=summary,
        data={
            "workflow_stage": stage,
            "search_task_id": state.get("paper_service_task_id"),
            "counts": {
                "discovered": progress.get("discovered", 0),
                "deduplicated": progress.get("deduplicated", 0),
                "existing_papers": progress.get("existing_in_database", 0),
                "new_papers": progress.get("new_candidates", 0),
                "recommended_papers": progress.get("recommendation_kept", 0),
                "persisted_relations": progress.get("persistence_saved", 0),
            },
            "supplemental_search_rounds": state.get(
                "supplemental_search_round",
                0,
            ),
            "source_stats": state.get("source_search_stats", {}),
            "warnings": state.get("warnings", []),
            "degraded": bool(state.get("degraded")),
            "pdf_cleanup_error": state.get("pdf_cleanup_error"),
            "remote_task_state": state.get("remote_task_state"),
            "task_status_update_error": state.get(
                "task_status_update_error"
            ),
        },
        artifact_refs=_artifact_refs(state),
        retryable=stage == "partial_failed",
        error_code=(
            None
            if stage in {"completed", "partial_failed"}
            else stage
        ),
        error_message=state.get("error"),
    )
