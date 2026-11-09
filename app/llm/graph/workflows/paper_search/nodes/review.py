import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphBubbleUp
from pydantic import BaseModel, ConfigDict, Field

from app.config import settings
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.provider import ValidatedJsonInvoker
from app.utils.search_adapters.contracts import SEARCH_PAGE_SIZE


class ProviderSupplementAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider_id: str
    requested_pages: int = Field(ge=1, le=2, strict=True)
    reason: str


class SearchReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["finish", "supplement"]
    provider_actions: list[ProviderSupplementAction] = Field(default_factory=list)
    reasoning: str


class ReviewNode:
    def __init__(self, *, model, artifact_store=None):
        self.model = model
        self.invoker = ValidatedJsonInvoker(model=model, schema=SearchReviewDecision)
        self.artifact_store = artifact_store or LocalArtifactStore()

    async def __call__(self, state):
        progress = state.get("progress", {})
        effective = progress.get("new_candidates", 0) + progress.get("existing_in_database", 0)
        round_number = state["supplemental_search_round"]
        eligible = {p: cursor for p, cursor in state.get("provider_cursors", {}).items() if cursor is not None}
        tasks = {}
        if (effective < settings.PAPER_SEARCH_TARGET_PAPER_COUNT and eligible
                and round_number < settings.PAPER_SEARCH_MAX_SUPPLEMENTAL_ROUNDS):
            try:
                decision = await self.invoker.ainvoke([
                    SystemMessage(content="审核检索结果。候选不足时从 eligible_providers 选择补搜来源，每来源1到2页，否则结束。只输出 SearchReviewDecision JSON。"),
                    HumanMessage(content=json.dumps({
                        "effective_count": effective, "target_count": settings.PAPER_SEARCH_TARGET_PAPER_COUNT,
                        "eligible_providers": list(eligible), "provider_stats": state.get("provider_stats", {}),
                    }, ensure_ascii=False)),
                ])
                decision = SearchReviewDecision.model_validate(decision)
                selected = [action.provider_id for action in decision.provider_actions]
                if (len(set(selected)) != len(selected) or any(p not in eligible for p in selected)
                        or (decision.action == "supplement") != bool(selected)):
                    raise ValueError("Invalid supplemental providers")
            except GraphBubbleUp:
                raise
            except Exception:
                # ponytail: a failed model review advances each eligible provider by one page.
                decision = SearchReviewDecision(action="supplement", reasoning="使用分页兜底策略。", provider_actions=[
                    ProviderSupplementAction(provider_id=p, requested_pages=1, reason="仍可续查") for p in eligible
                ])
            if decision.action == "supplement":
                for action in decision.provider_actions:
                    provider = action.provider_id
                    template = state["provider_query_plans"][provider]
                    pages = min(action.requested_pages, settings.PAPER_SEARCH_MAX_EXTRA_PAGES_PER_SOURCE)
                    tasks[provider] = {
                        **template, "round": round_number + 1, "cursor": eligible[provider],
                        "limit": SEARCH_PAGE_SIZE * pages, "max_pages": pages,
                    }
        else:
            decision = SearchReviewDecision(action="finish", reasoning="达到目标、轮数上限或没有可续查来源。")
        artifact = await self.artifact_store.write_json(
            run_id=state["child_run_id"], step_key="search_review", source="search_review",
            payload={"round": round_number, "effective_count": effective, "decision": decision.model_dump(mode="json")},
        )
        return {
            "stage": "searching" if tasks else "enriching", "status": "running",
            "active_search_tasks": tasks,
            "supplemental_search_round": round_number + 1 if tasks else round_number,
            "search_review_decision_artifact_ref": artifact.artifact_uri,
            "search_review_decision": decision.model_dump(mode="json"),
        }
