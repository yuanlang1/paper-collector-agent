from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.paper_search_schemas import PromptUnderstandingArgs
from app.llm.model_factory import create_validated_structured_chat_model


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
""".strip()


class PaperRecommendationResult(BaseModel):
    recommendation_stars: int = Field(ge=1, le=5)
    recommendation_reason: str = Field(
        min_length=1,
        max_length=1_000,
    )


class RecommendationNode:
    """根据检索意图、摘要与 venue 信息筛选当前任务的推荐论文。"""

    def __init__(
        self,
        artifact_store: LocalArtifactStore | None = None,
        model: Any | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.model = model or create_validated_structured_chat_model(
            PaperRecommendationResult,
            temperature=0,
        )

    async def _recommend_one(
        self,
        *,
        query_understanding: PromptUnderstandingArgs,
        paper: Mapping[str, Any],
    ) -> PaperRecommendationResult:
        paper_info = paper["paper_info"]

        model_input = {
            "query_understanding": query_understanding.model_dump(
                mode="json"
            ),
            "paper": {
                "title": paper_info.get("title"),
                "authors": paper_info.get("authors"),
                "paper_abstract": paper_info.get(
                    "paper_abstract"
                ),
                "ai_abstract": paper_info.get("ai_abstract"),
                "keywords": paper_info.get("keywords"),
                "source": paper_info.get("source"),
            },
            "venue": paper.get("venue_resolution"),
        }

        result = await self.model.ainvoke(
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

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            run_id = state.get("run_id")
            abstract_artifact_uri = state.get(
                "abstract_manifest_artifact_ref"
            )
            existing_artifact_uri = state.get(
                "existing_papers_manifest_artifact_ref"
            )

            if not isinstance(run_id, str) or not run_id:
                raise ValueError("缺少有效 run_id。")

            artifact_inputs = [
                ("abstract", abstract_artifact_uri),
                ("existing", existing_artifact_uri),
            ]
            valid_artifact_inputs = [
                (name, uri)
                for name, uri in artifact_inputs
                if uri is not None
            ]

            if not valid_artifact_inputs:
                raise ValueError(
                    "缺少 abstract 或 existing papers manifest artifact。"
                )

            for name, uri in valid_artifact_inputs:
                if (
                    not isinstance(uri, str)
                    or not uri.startswith("artifact://")
                ):
                    raise ValueError(
                        f"{name} manifest artifact 格式无效。"
                    )

            query_understanding = PromptUnderstandingArgs.model_validate(
                state.get("query_understanding")
            )

            base_dir = self.artifact_store.base_dir.resolve()

            def read_manifest(
                name: str,
                artifact_uri: str,
            ) -> dict[str, Any]:
                manifest_path = (
                    base_dir
                    / artifact_uri.removeprefix("artifact://")
                ).resolve()

                try:
                    manifest_path.relative_to(base_dir)
                except ValueError as exc:
                    raise ValueError(
                        f"{name} manifest 超出 artifact 存储目录。"
                    ) from exc

                with manifest_path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    manifest = json.load(file)

                if not isinstance(manifest, dict):
                    raise ValueError(
                        f"{name} manifest 内容格式无效。"
                    )

                if not isinstance(manifest.get("papers", []), list):
                    raise ValueError(
                        f"{name} manifest papers 格式无效。"
                    )

                return manifest

            manifests = await asyncio.gather(
                *[
                    asyncio.to_thread(read_manifest, name, uri)
                    for name, uri in valid_artifact_inputs
                ]
            )
            manifests_by_name = dict(
                zip(
                    [name for name, _ in valid_artifact_inputs],
                    manifests,
                    strict=True,
                )
            )
            abstract_manifest = manifests_by_name.get("abstract")
            existing_manifest = manifests_by_name.get("existing")
            new_papers = (
                abstract_manifest.get("papers", [])
                if abstract_manifest
                else []
            )
            existing_papers = (
                existing_manifest.get("papers", [])
                if existing_manifest
                else []
            )
            manifest = dict(abstract_manifest or existing_manifest or {})

            kept_papers: list[dict[str, Any]] = []
            removed_low_recommendation: list[dict[str, Any]] = []
            removed_recommendation_failed: list[dict[str, str]] = []
            skipped_invalid_existing: list[dict[str, str]] = []
            warnings: list[str] = []

            recommend_input: list[dict[str, Any]] = []

            for paper in new_papers:
                if not isinstance(paper, dict):
                    continue

                paper["paper_id"] = None
                recommend_input.append(paper)

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
                            "reason": (
                                "已有论文缺少合法 paper_id 或 paper_info。"
                            ),
                        }
                    )
                    continue

                recommend_input.append(paper)

            for paper in recommend_input:
                paper_info = (
                    paper.get("paper_info")
                    if isinstance(paper, dict)
                    else None
                )

                if not isinstance(paper_info, dict):
                    continue

                try:
                    recommendation = await self._recommend_one(
                        query_understanding=query_understanding,
                        paper=paper,
                    )
                except Exception as exc:
                    title = str(paper_info.get("title") or "")

                    removed_recommendation_failed.append(
                        {
                            "title": title,
                            "reason": f"推荐评分失败：{exc}",
                        }
                    )
                    warnings.append(
                        f"{title} 的推荐评分失败，已从当前任务结果移除。"
                    )
                    continue

                relation_draft = {
                    "search_task_id": state.get(
                        "paper_service_task_id"
                    ),
                    "paper_id": paper.get("paper_id"),
                    "recommendation_stars": (
                        recommendation.recommendation_stars
                    ),
                    "recommendation_reason": (
                        recommendation.recommendation_reason
                    ),
                }

                if recommendation.recommendation_stars < 3:
                    removed_low_recommendation.append(
                        {
                            "title": paper_info.get("title"),
                            **relation_draft,
                        }
                    )
                    continue

                paper["task_paper_relation_draft"] = relation_draft
                kept_papers.append(paper)

            manifest["papers"] = kept_papers
            manifest["removed_low_recommendation"] = (
                removed_low_recommendation
            )
            manifest["removed_recommendation_failed"] = (
                removed_recommendation_failed
            )
            manifest["skipped_invalid_existing"] = (
                skipped_invalid_existing
            )
            manifest["step_key"] = "recommend_papers"

            artifact = await self.artifact_store.write_json(
                run_id=run_id,
                step_key="recommend_papers",
                source="recommendation",
                kind="paper_info_recommendation_manifest_json",
                payload=manifest,
                count=len(kept_papers),
                metadata={
                    "abstract_input_manifest": abstract_artifact_uri,
                    "existing_input_manifest": existing_artifact_uri,
                    "new_input_count": len(new_papers),
                    "existing_input_count": len(existing_papers),
                    "kept_count": len(kept_papers),
                    "low_recommendation_count": len(
                        removed_low_recommendation
                    ),
                    "recommendation_failed_count": len(
                        removed_recommendation_failed
                    ),
                    "invalid_existing_count": len(
                        skipped_invalid_existing
                    ),
                },
            )

        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"论文推荐失败：{exc}",
            }

        recommendation_failed = bool(
            removed_recommendation_failed
            or skipped_invalid_existing
        )
        degraded = bool(state.get("degraded")) or recommendation_failed

        if not kept_papers:
            if recommendation_failed and not removed_low_recommendation:
                terminal_stage = "failed"
            elif degraded:
                terminal_stage = "partial_failed"
            else:
                terminal_stage = "completed"
            empty_warning = (
                "所有候选论文的推荐评分均失败。"
                if terminal_stage == "failed"
                else "没有推荐星数大于或等于 3 的论文。"
            )
            terminal_error = {
                "completed": None,
                "partial_failed": (
                    "recommendation candidates were partially unavailable"
                ),
                "failed": "all recommendation candidates failed",
            }[terminal_stage]
            return {
                "stage": terminal_stage,
                "status": terminal_stage,
                "degraded": degraded,
                "recommendation_manifest_artifact_ref": (
                    artifact.artifact_uri
                ),
                "progress": {
                    **state.get("progress", {}),
                    "recommendation_kept": 0,
                    "recommendation_removed": len(
                        removed_low_recommendation
                    ),
                    "recommendation_failed": len(
                        removed_recommendation_failed
                    ),
                    "recommendation_invalid_existing": len(
                        skipped_invalid_existing
                    ),
                },
                "warnings": [
                    *state.get("warnings", []),
                    *warnings,
                    *(
                        [
                            (
                                f"已跳过 {len(skipped_invalid_existing)} "
                                "条无效的已有论文记录。"
                            )
                        ]
                        if skipped_invalid_existing
                        else []
                    ),
                    empty_warning,
                ],
                "error": terminal_error,
            }

        return {
            "stage": "persisting_papers",
            "status": (
                "partial_failed"
                if degraded
                else "running"
            ),
            "degraded": degraded,
            "recommendation_manifest_artifact_ref": artifact.artifact_uri,
            "progress": {
                **state.get("progress", {}),
                "recommendation_kept": len(kept_papers),
                "recommendation_removed": len(
                    removed_low_recommendation
                ),
                "recommendation_failed": len(
                    removed_recommendation_failed
                ),
                "recommendation_invalid_existing": len(
                    skipped_invalid_existing
                ),
            },
            "warnings": [
                *state.get("warnings", []),
                *warnings,
                *(
                    [
                        (
                            f"已跳过 {len(skipped_invalid_existing)} "
                            "条无效的已有论文记录。"
                        )
                    ]
                    if skipped_invalid_existing
                    else []
                ),
            ],
            "error": None,
        }
