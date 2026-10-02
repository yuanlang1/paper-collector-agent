from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import settings
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.provider import ChatClient, ModelOptions


SEARCH_REVIEW_SYSTEM_PROMPT = """
你是论文检索质量审核员，负责根据当前候选数量和 Google Scholar 分页状态决定结束检索或补充检索。

输入包含 effective_count、target_count、source_stats 和 allowed_sources。

规则：
- 只能从 allowed_sources 中选择来源；当前只允许 Google Scholar。
- 仅允许 next_page 策略，且只选择 source_stats.has_next 为 true 的来源。
- 候选数量已达到目标或无可靠补充路径时，选择 finish。
- 补充时 requested_pages 为 1 或 2，priority 越小优先级越高。
- 只依据输入做判断，不得虚构候选数量、来源状态或检索结果。

输出规范：
- 只输出合法 JSON 对象，不要输出 Markdown、代码块或额外说明。
- action 为 finish 时 source_actions 必须为空列表；action 为 supplement 时至少包含一个合法动作。

结束检索样例：
{
  "action": "finish",
  "required_additional_count": 0,
  "source_actions": [],
  "reasoning": "有效候选数量已达到目标。"
}

补充检索样例：
{
  "action": "supplement",
  "required_additional_count": 5,
  "source_actions": [
    {
      "source": "Google Scholar",
      "strategy": "next_page",
      "requested_pages": 1,
      "priority": 1,
      "reason": "当前候选数量不足，且 Google Scholar 仍有下一页。"
    }
  ],
  "reasoning": "继续获取下一页结果以补足候选论文。"
}
""".strip()


class SourceSupplementAction(BaseModel):
    source: Literal["Google Scholar"]
    strategy: Literal["next_page"]
    requested_pages: int = Field(ge=1, le=2)
    priority: int = Field(ge=1, le=10)
    reason: str = Field(min_length=1, max_length=500)


class SearchReviewDecision(BaseModel):
    action: Literal["finish", "supplement"]
    required_additional_count: int = Field(ge=0)
    source_actions: list[SourceSupplementAction] = Field(default_factory=list)
    reasoning: str = Field(min_length=1, max_length=1000)


class ReviewNode:
    def __init__(
        self,
        artifact_store: LocalArtifactStore | None = None,
        model: Any | None = None,
        chat: ChatClient | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        client = chat or ChatClient()
        self.model = model or client.structured(
            SearchReviewDecision,
            options=ModelOptions(temperature=0),
        )

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        child_run_id = state["child_run_id"]

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
                    "requested_sources", ["Google Scholar"]
                ),
            }
            try:
                result = await self.model.ainvoke([
                    SystemMessage(content=SEARCH_REVIEW_SYSTEM_PROMPT),
                    HumanMessage(
                        content=json.dumps(prompt, ensure_ascii=False, default=str)
                    ),
                ])
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
                run_id=child_run_id,
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

        update = {
            "search_review_decision_artifact_ref": artifact.artifact_uri,
            "search_review_decision": decision.model_dump(mode="json"),
        }
        if decision.action == "finish":
            return {**update, "stage": "enriching", "status": "running"}

        return self._plan_supplement(state, decision, update)

    def _plan_supplement(
        self,
        state: Mapping[str, Any],
        decision: SearchReviewDecision,
        update: dict[str, Any],
    ) -> dict[str, Any]:
        cursors = state.get("source_search_cursors") or {}
        plans_by_source = {
            plan.get("source"): plan
            for plan in state.get("source_query_plans", [])
            if isinstance(plan, dict)
        }
        action = next(
            (
                item
                for item in sorted(
                    decision.model_dump(mode="json").get("source_actions", []),
                    key=lambda value: value.get("priority", 10),
                )
                if item.get("source") == "Google Scholar"
            ),
            None,
        )
        cursor = cursors.get("Google Scholar", {})
        plan_template = plans_by_source.get("Google Scholar")
        if (
            not isinstance(action, dict)
            or action.get("strategy") != "next_page"
            or not cursor.get("has_next")
            or not isinstance(plan_template, dict)
        ):
            return {
                **update,
                "stage": "enriching",
                "status": "partial_failed",
                "degraded": True,
                "warnings": [
                    *state.get("warnings", []),
                    "无可用补充检索来源，继续后续流程。",
                ],
            }

        plan = {
            **plan_template,
            "arguments": dict(plan_template["arguments"]),
        }
        plan["arguments"]["start"] = cursor["next_offset"]
        requested_pages = min(
            int(action.get("requested_pages", 1)),
            settings.PAPER_SEARCH_MAX_EXTRA_PAGES_PER_SOURCE,
        )
        plan["arguments"]["total_limit"] = (
            int(plan["arguments"]["num"]) * requested_pages
        )
        plan["arguments"]["max_pages"] = requested_pages
        return {
            **update,
            "stage": "searching",
            "status": "running",
            "active_search_sources": ["Google Scholar"],
            "active_source_query_plans": [plan],
            "supplemental_search_round": (
                int(state.get("supplemental_search_round", 0)) + 1
            ),
        }
