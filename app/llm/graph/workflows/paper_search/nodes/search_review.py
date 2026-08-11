from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import settings
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.model_factory import create_validated_structured_chat_model


class SourceSupplementAction(BaseModel):
    source: Literal["arXiv", "DBLP", "Google Scholar"]
    strategy: Literal["next_page"]
    requested_pages: int = Field(ge=1, le=2)
    priority: int = Field(ge=1, le=10)
    reason: str = Field(min_length=1, max_length=500)


class SearchReviewDecision(BaseModel):
    action: Literal["finish", "supplement"]
    required_additional_count: int = Field(ge=0)
    source_actions: list[SourceSupplementAction] = Field(default_factory=list)
    reasoning: str = Field(min_length=1, max_length=1000)


class SearchReviewBrainNode:
    def __init__(
        self, artifact_store: LocalArtifactStore | None = None, model: Any | None = None
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.model = model or create_validated_structured_chat_model(
            SearchReviewDecision, temperature=0
        )

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        run_id = state.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return {"stage": "failed", "status": "failed", "error": "missing run_id"}
        progress = state.get("progress", {})
        effective = int(progress.get("new_candidates", 0)) + int(
            progress.get("existing_in_database", 0)
        )
        round_number = int(state.get("supplemental_search_round", 0))
        stats = state.get("source_search_stats", {})
        target = settings.PAPER_SEARCH_TARGET_PAPER_COUNT
        if (
            effective >= target
            or round_number >= settings.PAPER_SEARCH_MAX_SUPPLEMENTAL_ROUNDS
        ):
            decision = SearchReviewDecision(
                action="finish",
                required_additional_count=max(target - effective, 0),
                reasoning="候选论文数量已满足目标或已达到补充检索轮次上限。",
            )
        else:
            prompt = {
                "effective_count": effective,
                "target_count": target,
                "source_stats": stats,
                "allowed_sources": state.get(
                    "requested_sources", ["arXiv", "DBLP", "Google Scholar"]
                ),
            }
            try:
                result = await self.model.ainvoke(
                    [
                        SystemMessage(
                            content="审核论文检索质量。仅从 allowed_sources 选择来源；只允许 next_page，且必须选择仍有下一页的来源。"
                        ),
                        HumanMessage(
                            content=json.dumps(prompt, ensure_ascii=False, default=str)
                        ),
                    ]
                )
                decision = (
                    result
                    if isinstance(result, SearchReviewDecision)
                    else SearchReviewDecision.model_validate(result)
                )
            except Exception:
                candidates = [
                    source for source, stat in stats.items() if stat.get("has_next")
                ]
                decision = SearchReviewDecision(
                    action="supplement" if candidates else "finish",
                    required_additional_count=max(target - effective, 0),
                    source_actions=[
                        SourceSupplementAction(
                            source=source,
                            strategy="next_page",
                            requested_pages=1,
                            priority=index + 1,
                            reason="模型审核不可用，使用分页兜底策略。",
                        )
                        for index, source in enumerate(candidates[:1])
                    ],
                    reasoning="使用确定性兜底策略。",
                )
        try:
            artifact = await self.artifact_store.write_json(
                run_id=run_id,
                step_key="search_review",
                source="search_review",
                kind="search_review_decision_json",
                payload={
                    "effective_count": effective,
                    "target_count": target,
                    "round": round_number,
                    "source_stats": stats,
                    "decision": decision.model_dump(mode="json"),
                },
            )
        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"检索质量审核结果保存失败：{exc}",
            }
        return {
            "stage": "planning_supplemental_search"
            if decision.action == "supplement"
            else "enriching",
            "status": "running",
            "search_review_decision_artifact_ref": artifact.artifact_uri,
            "search_review_decision": decision.model_dump(mode="json"),
        }
