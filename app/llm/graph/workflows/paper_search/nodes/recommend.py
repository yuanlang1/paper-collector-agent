from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphBubbleUp

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.paper_search.paper_search_schemas import (
    PaperRecommendationResult,
    PromptUnderstandingArgs,
)
from app.llm.provider import ValidatedJsonInvoker

RECOMMENDATION_SYSTEM_PROMPT = """
你负责评估论文与用户检索意图的匹配程度。

输入包含：
- 用户已确认的 query_understanding；
- 论文标题、原始摘要、中文 AI 摘要；
- 已解析的 venue 信息。

请输出 1 至 5 的整数推荐星数及中文推荐理由。

评分标准：
- 5：研究主题、目标、限制条件均高度匹配；
- 4：核心主题高度匹配，存在少量范围差异；
- 3：满足基本检索需求，具有参考价值；
- 2：存在部分关联，但偏离用户核心需求；
- 1：关联很弱或明显不符合检索意图。

规则：
- 只能依据输入内容评分，不得编造论文贡献、实验结果或 venue 等级；
- venue 信息仅作为辅助判断，不得因缺失而臆测；
- 推荐理由使用中文，明确说明匹配点或不匹配点。

输出规范：
- 只输出合法 JSON 对象，不要输出 Markdown、代码块或额外说明。
- recommendation_stars 必须是 1 至 5 的整数，recommendation_reason 必须为中文。

输出样例：
{
  "recommendation_stars": 4,
  "recommendation_reason": "论文聚焦检索增强生成方法，与用户主题高度匹配，但未覆盖全部限定条件。"
}
"""


class RecommendationNode:
    def __init__(
        self,
        *,
        model: BaseChatModel,
    ) -> None:
        self.model: BaseChatModel = model
        self.invoker = ValidatedJsonInvoker(
            model=self.model,
            schema=PaperRecommendationResult,
        )

    async def _recommend_one(
        self,
        *,
        query_understanding: PromptUnderstandingArgs,
        paper: Mapping[str, Any],
    ) -> PaperRecommendationResult:
        paper_info = paper["paper_info"]

        model_input = {
            "query_understanding": query_understanding.model_dump(mode="json"),
            "paper": {
                "title": paper_info.get("title"),
                "authors": paper_info.get("authors"),
                "paper_abstract": paper_info.get("paper_abstract"),
                "ai_abstract": paper_info.get("ai_abstract"),
                "keywords": paper_info.get("keywords"),
                "source": paper_info.get("source"),
            },
            "venue": paper.get("venue_resolution"),
        }

        result = await self.invoker.ainvoke(
            [
                SystemMessage(
                    content=RECOMMENDATION_SYSTEM_PROMPT,
                ),
                HumanMessage(
                    content=json.dumps(
                        model_input,
                        ensure_ascii=False,
                        default=str,
                    )
                ),
            ]
        )

        return (
            result
            if isinstance(result, PaperRecommendationResult)
            else PaperRecommendationResult.model_validate(result)
        )

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        try:
            result = await self._recommend_one(
                query_understanding=PromptUnderstandingArgs.model_validate(
                    state["query_understanding"]
                ),
                paper=state["paper"],
            )
            score = result.model_dump(mode="json")
        except GraphBubbleUp:
            raise
        except Exception as exc:
            score = {"error": str(exc)}
        return {"recommendation_results": {state["paper_index"]: score}}


class PrepareRecommendationNode:
    def __init__(self, artifact_store: LocalArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()

    @staticmethod
    def _prepare_papers(
        new_papers: list[dict[str, Any]],
        existing_papers: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        skipped_invalid_existing: list[dict[str, str]] = []
        recommend_input: list[dict[str, Any]] = []

        for paper in new_papers:
            recommend_input.append({**paper, "paper_id": None})

        for paper in existing_papers:
            if not isinstance(paper, dict):
                skipped_invalid_existing.append(
                    {
                        "title": "",
                        "reason": "已有论文记录不是对象。",
                    }
                )
                continue

            paper_info = paper.get("paper_info")
            paper_id = paper.get("paper_id")

            if (
                not isinstance(paper_info, dict)
                or not isinstance(paper_id, int)
                or isinstance(paper_id, bool)
                or paper_id <= 0
            ):
                skipped_invalid_existing.append(
                    {
                        "title": (
                            str(paper_info.get("title") or "")
                            if isinstance(paper_info, dict)
                            else ""
                        ),
                        "reason": ("已有论文缺少合法 paper_id 或 paper_info。"),
                    }
                )
                continue

            recommend_input.append(paper)
        return recommend_input, skipped_invalid_existing

    async def _load_papers(
        self,
        enriched_artifact_uri: str | None,
        existing_artifact_uri: str | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        if enriched_artifact_uri is None and existing_artifact_uri is None:
            raise ValueError("缺少补全论文或已有论文清单。")

        enriched_manifest = (
            await self.artifact_store.read_json_uri(enriched_artifact_uri)
            if enriched_artifact_uri is not None
            else None
        )
        existing_manifest = (
            await self.artifact_store.read_json_uri(existing_artifact_uri)
            if existing_artifact_uri is not None
            else None
        )
        new_papers = enriched_manifest.get("papers", []) if enriched_manifest else []
        existing_papers = (
            existing_manifest.get("papers", []) if existing_manifest else []
        )
        manifest = dict(enriched_manifest or existing_manifest or {})
        return manifest, new_papers, existing_papers

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        try:
            query = PromptUnderstandingArgs.model_validate(
                state.get("query_understanding")
            )
            manifest, new_papers, existing_papers = await self._load_papers(
                state.get("enriched_manifest_artifact_ref"),
                state.get("existing_papers_manifest_artifact_ref"),
            )
            papers, skipped = self._prepare_papers(new_papers, existing_papers)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"论文推荐失败：{exc}",
            }

        return {
            "query_understanding": query.model_dump(mode="json"),
            "recommendation_candidates": papers,
            "recommendation_manifest": {
                **manifest,
                "papers": [],
                "skipped_invalid_existing": skipped,
            },
        }


class CollectRecommendationNode:
    def __init__(self, artifact_store: LocalArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()

    @staticmethod
    def _collect_scores(
        recommend_input: list[dict[str, Any]],
        results: Mapping[int, dict[str, Any]],
        task_id: int | None,
    ) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]], list[str]
    ]:
        kept_papers = []
        removed_low_recommendation = []
        removed_recommendation_failed = []
        warnings = []
        for index, paper in enumerate(recommend_input):
            result = results[index]
            paper_info = paper.get("paper_info") or {}

            if "error" in result:
                title = str(paper_info.get("title") or "")

                removed_recommendation_failed.append(
                    {
                        "title": title,
                        "reason": f"推荐评分失败：{result['error']}",
                    }
                )
                warnings.append(f"{title} 的推荐评分失败，已从当前任务结果移除。")
                continue

            relation_draft = {
                "search_task_id": task_id,
                "paper_id": paper.get("paper_id"),
                "recommendation_stars": result["recommendation_stars"],
                "recommendation_reason": result["recommendation_reason"],
            }

            if relation_draft["recommendation_stars"] < 3:
                removed_low_recommendation.append(
                    {
                        "title": paper_info.get("title"),
                        **relation_draft,
                    }
                )
                continue

            kept_papers.append({**paper, "task_paper_relation_draft": relation_draft})
        return (
            kept_papers,
            removed_low_recommendation,
            removed_recommendation_failed,
            warnings,
        )

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            child_run_id = state["child_run_id"]
            manifest = dict(state["recommendation_manifest"])
            skipped_invalid_existing = manifest["skipped_invalid_existing"]

            (
                kept_papers,
                removed_low_recommendation,
                removed_recommendation_failed,
                warnings,
            ) = self._collect_scores(
                state["recommendation_candidates"],
                state.get("recommendation_results", {}),
                state.get("paper_service_task_id"),
            )

            manifest["papers"] = kept_papers
            manifest["removed_low_recommendation"] = removed_low_recommendation
            manifest["removed_recommendation_failed"] = removed_recommendation_failed
            manifest["step_key"] = "recommend_papers"

            artifact = await self.artifact_store.write_json(
                run_id=child_run_id,
                step_key="recommend_papers",
                source="recommendation",
                payload=manifest,
            )

        except GraphBubbleUp:
            raise
        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"论文推荐失败：{exc}",
            }

        if skipped_invalid_existing:
            warnings.append(
                f"已跳过 {len(skipped_invalid_existing)} 条无效的已有论文记录。"
            )
        if not kept_papers:
            warnings.append(
                "所有候选论文的推荐评分均失败。"
                if (removed_recommendation_failed or skipped_invalid_existing)
                and not removed_low_recommendation
                else "没有推荐星数大于或等于 3 的论文。"
            )
        return {
            "stage": "persisting_papers" if kept_papers else "failed",
            "status": "running" if kept_papers else "failed",
            "recommendation_manifest_artifact_ref": artifact.artifact_uri,
            "progress": {
                **state.get("progress", {}),
                "recommendation_kept": len(kept_papers),
                "recommendation_removed": len(removed_low_recommendation),
                "recommendation_failed": len(removed_recommendation_failed),
                "recommendation_invalid_existing": len(skipped_invalid_existing),
            },
            "warnings": [*state.get("warnings", []), *warnings],
            "error": None if kept_papers else "没有可用的推荐论文。",
        }
