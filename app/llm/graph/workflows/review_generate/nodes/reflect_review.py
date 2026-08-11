from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.model_factory import create_validated_structured_chat_model


REF_PATTERN = re.compile(r"\[\[REF_([^\]]+)\]\]")


class ReflectionIssue(BaseModel):
    severity: Literal["critical", "major", "minor"]
    category: str = "uncategorized"
    description: str


class ReflectionResult(BaseModel):
    satisfied: bool
    summary: str = ""
    issues: list[ReflectionIssue] = Field(default_factory = list)

    retrieve_claim_ids: list[str] = Field(default_factory = list)
    revise_claim_ids: list[str] = Field(default_factory = list)
    render_section_ids: list[str] = Field(default_factory = list)

    @model_validator(mode="after")
    def provide_summary_when_missing(self) -> "ReflectionResult":
        if self.summary.strip():
            return self

        if self.issues:
            self.summary = "；".join(
                issue.description
                for issue in self.issues
            )
        else:
            self.summary = (
                "Review passed." if self.satisfied else "Review requires revision."
            )
        return self


REFLECT_REVIEW_PROMPT = """
    你是一位严格的学术综述审稿人。

    请评估当前综述是否达到可交付标准，并在不合格时给出最小修订范围。

    你只能依据输入中的 Framework、综述草稿、系统硬约束检查结果、
    Claim 列表和证据概览作出判断。

    规则：
    1. 不得建议外部检索、Auto Search、扩大任务论文集合。
    2. 不得引入新论文、新事实、新作者、新引用或未提供的研究结论。
    3. system_hard_issues 非空时，satisfied 必须为 false。
    4. 存在 critical 或 major 问题时，satisfied 必须为 false。
    5. 需要更多任务内正文证据时，将 Claim ID 放入 retrieve_claim_ids。
    6. Claim 本身过宽、过强或不应保留时，将其 ID 放入 revise_claim_ids。
    7. 需要修改论述、引用绑定、章节连贯性时，将章节 ID 放入 render_section_ids。
    8. 只选择实际需要修订的目标，不要重写无关章节。
    9. 仅返回结构化输出，且必须包含所有字段。即使没有问题，也要返回空数组；
       每个 issue 必须包含 severity、category 和 description。

    输出格式：
    {
      "satisfied": false,
      "summary": "简要说明是否达到交付标准",
      "issues": [
        {
          "severity": "major",
          "category": "claim_evidence_mismatch",
          "description": "问题说明"
        }
      ],
      "retrieve_claim_ids": [],
      "revise_claim_ids": ["claim_001"],
      "render_section_ids": ["section_id"]
    }
""".strip()


class ReflectReviewNode:
    def __init__(
        self,
        *,
        artifact_store: LocalArtifactStore | None = None,
        model: Any | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.model = model or create_validated_structured_chat_model(
            ReflectionResult,
            temperature = 0,
        )

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            (
                framework_payload,
                claims_payload,
                evidence_payload,
                corpus_payload,
                review_draft,
            ) = await asyncio.gather(
                self.artifact_store.read_json_uri(
                    state["framework_artifact_ref"]
                ),
                self.artifact_store.read_json_uri(
                    state["claims_artifact_ref"]
                ),
                self.artifact_store.read_json_uri(
                    state["evidence_ledger_artifact_ref"]
                ),
                self.artifact_store.read_json_uri(
                    state["corpus_artifact_ref"]
                ),
                self.artifact_store.read_json_uri(
                    state["review_draft_artifact_ref"]
                ),
            )

            hard_issues = self._check_hard_constraints(
                review_draft = review_draft,
                corpus_payload = corpus_payload,
            )

            result = await self.model.ainvoke(
                [
                    SystemMessage(content = REFLECT_REVIEW_PROMPT),
                    HumanMessage(
                        content = json.dumps(
                            {
                                "topic": state["topic"],
                                "language": state["language"],
                                "review_type": state["review_type"],
                                "framework": framework_payload["framework"],
                                "review_draft": review_draft,
                                "system_hard_issues": hard_issues,
                                "claims": self._flatten_claims(
                                    claims_payload
                                ),
                                "evidence": self._evidence_overview(
                                    evidence_payload
                                ),
                            },
                            ensure_ascii = False,
                        )
                    ),
                ]
            )

            reflection = (
                result
                if isinstance(result, ReflectionResult)
                else ReflectionResult.model_validate(result)
            )
        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"review reflection failed: {exc}",
            }

        claim_ids = {
            claim["claim_id"]
            for claim in self._flatten_claims(claims_payload)
        }
        section_ids = {
            section["section_id"]
            for section in framework_payload["framework"]["sections"]
        }

        retrieve_claim_ids = self._valid_ids(
            reflection.retrieve_claim_ids,
            claim_ids,
        )
        revise_claim_ids = self._valid_ids(
            reflection.revise_claim_ids,
            claim_ids,
        )
        render_section_ids = self._valid_ids(
            reflection.render_section_ids,
            section_ids,
        )

        blocking_issues = [
            issue
            for issue in reflection.issues
            if issue.severity in {"critical", "major"}
        ]
        satisfied = (
            reflection.satisfied
            and not hard_issues
            and not blocking_issues
        )

        current_round = state["reflection_round"]
        next_round = current_round + 1

        reflection_report = {
            "task_id": state["task_id"],
            "reflection_round": current_round,
            "satisfied": satisfied,
            "summary": reflection.summary,
            "hard_issues": hard_issues,
            "issues": [
                issue.model_dump(mode = "json")
                for issue in reflection.issues
            ],
            "revision_targets": {
                "retrieve_claim_ids": retrieve_claim_ids,
                "revise_claim_ids": revise_claim_ids,
                "render_section_ids": render_section_ids,
            },
        }

        artifact = await self.artifact_store.write_json(
            run_id = state["run_id"],
            step_key = f"reflect_review_{current_round}",
            source = "task_review",
            kind = "task_review_reflection_report_json",
            count = len(hard_issues) + len(reflection.issues),
            payload = reflection_report,
        )

        if satisfied:
            return {
                "reflection_report_artifact_ref": artifact.artifact_uri,
                "revision_plan_artifact_ref": None,
                "stage": "finalizing_handoff",
                "status": "running",
                "error": None,
            }

        revision_plan = {
            "task_id": state["task_id"],
            "reflection_round": next_round,
            "retrieve_claim_ids": retrieve_claim_ids,
            "revise_claim_ids": revise_claim_ids,
            "render_section_ids": render_section_ids,
        }

        revision_artifact = await self.artifact_store.write_json(
            run_id = state["run_id"],
            step_key = f"revision_plan_{next_round}",
            source = "task_review",
            kind = "task_review_revision_plan_json",
            count = (
                len(retrieve_claim_ids)
                + len(revise_claim_ids)
                + len(render_section_ids)
            ),
            payload = revision_plan,
        )

        if next_round >= state["max_reflection_rounds"]:
            return {
                "reflection_round": next_round,
                "reflection_report_artifact_ref": artifact.artifact_uri,
                "revision_plan_artifact_ref": revision_artifact.artifact_uri,
                "stage": "failed",
                "status": "failed",
                "error": "review did not satisfy reflection criteria",
            }

        if revise_claim_ids:
            next_stage = "generating_claims"
        elif retrieve_claim_ids:
            next_stage = "retrieving_evidence"
        elif render_section_ids:
            next_stage = "rendering_sections"
        else:
            return {
                "reflection_round": next_round,
                "reflection_report_artifact_ref": artifact.artifact_uri,
                "revision_plan_artifact_ref": revision_artifact.artifact_uri,
                "stage": "failed",
                "status": "failed",
                "error": "reflection produced no executable revision target",
            }

        return {
            "reflection_round": next_round,
            "reflection_report_artifact_ref": artifact.artifact_uri,
            "revision_plan_artifact_ref": revision_artifact.artifact_uri,
            "retrieve_claim_ids": retrieve_claim_ids,
            "revise_claim_ids": revise_claim_ids,
            "render_section_ids": render_section_ids,
            "stage": next_stage,
            "status": "running",
            "error": None,
        }

    def _check_hard_constraints(
        self,
        *,
        review_draft: dict[str, Any],
        corpus_payload: dict[str, Any],
    ) -> list[str]:
        text = "\n".join(
            [
                review_draft.get("abstract", ""),
                review_draft.get("body_markdown", ""),
                review_draft.get("conclusion", ""),
            ]
        )

        anchors = REF_PATTERN.findall(text)
        known_paper_ids = {
            str(paper["paper_id"])
            for paper in corpus_payload["papers"]
        }

        issues = []

        if not review_draft.get("body_markdown", "").strip():
            issues.append("review body is empty")

        if not review_draft.get("abstract", "").strip():
            issues.append("review abstract is empty")

        if not review_draft.get("conclusion", "").strip():
            issues.append("review conclusion is empty")

        if not anchors:
            issues.append("review has no citation anchors")

        for paper_id in anchors:
            if not paper_id.isdigit():
                issues.append(f"malformed citation anchor: REF_{paper_id}")
            elif paper_id not in known_paper_ids:
                issues.append(f"citation is outside task corpus: REF_{paper_id}")

        return list(dict.fromkeys(issues))

    def _flatten_claims(
        self,
        claims_payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return [
            claim
            for section in claims_payload["claims"]["sections"]
            for claim in section["claims"]
        ]

    def _evidence_overview(
        self,
        evidence_payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return [
            {
                "claim_id": item["claim_id"],
                "section_id": item["section_id"],
                "status": item["status"],
                "chunk_count": len(item["chunk_snippets"]),
                "paper_ids": list(
                    dict.fromkeys(
                        snippet["paper_id"]
                        for snippet in item["chunk_snippets"]
                    )
                ),
            }
            for item in evidence_payload["claims"]
        ]

    def _valid_ids(
        self,
        values: list[str],
        allowed: set[str],
    ) -> list[str]:
        return list(
            dict.fromkeys(
                value
                for value in values
                if value in allowed
            )
        )

