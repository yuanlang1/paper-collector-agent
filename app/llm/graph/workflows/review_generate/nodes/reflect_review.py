from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.citations import (
    citation_anchor_ids,
    unsupported_synthesis_anchor_ids,
)
from app.llm.graph.workflows.review_generate.contracts import invoke, save
from app.llm.graph.workflows.review_generate.nodes.finalize import failed
from app.llm.provider import ChatClient, ModelOptions


WRITING_REVIEW_PROMPT = """
    你是学术综述写作质量控制编辑。依据既定 Framework、已核验 Claim 与当前草稿，
    检查章节是否回应 discussion_questions、跨研究综合是否清楚、衔接和术语是否一致、
    限定条件是否被保留，以及摘要、正文和结论是否一致。

    只提出影响理解或结论准确性的必要写作修改。修改目标只能是 section 或 synthesis；
    保持已有研究范围、论点和证据基础，不得要求新增 Claim、重建 Framework 或补充检索。
    证据本身不足应说明为受阻原因，不能伪装成可由写作解决的问题。一般润色写入 suggestions，
    不触发修改。程序提供的 hard_issues 未解决时不得 decision=pass。

    review_mode 为 initial 时进行一次全文检查；为 acceptance 时，重点验收 previous_plan 的问题是否解决，并检查修改是否引入明显不一致。

    以下仅为格式示例；使用实际 section_id 和问题，勿照抄：
    pass：
    ```json
    {
    "decision": "pass",
    "summary": "必要写作问题已解决。",
    "revisions": [],
    "suggestions": ["可统一术语表述。"]
    }
    ```
    revise：
    ```json
    {
    "decision": "revise",
    "summary": "章节衔接与摘要结论仍不一致。",
    "revisions": [
        {
        "target_type": "section",
        "target_id": "methods",
        "issue": "未回应章节讨论问题。",
        "required_change": "补充对已核验条件差异的综合。",
        "acceptance_criteria": "正文明确比较已核验的条件差异且不新增结论。"
        },
        {
        "target_type": "synthesis",
        "target_id": "review",
        "issue": "摘要扩大了正文结论。",
        "required_change": "收缩为正文已有的限定表述。",
        "acceptance_criteria": "摘要与正文的范围和强度一致。"
        }
    ],
    "suggestions": []
    }
    ```
    blocked：
    ```json
    {
    "decision": "blocked",
    "summary": "正文缺少可支撑必要结论的已核验证据，无法仅靠写作解决。",
    "revisions": [],
    "suggestions": []
    }
    ```
""".strip()


class WritingRevision(BaseModel):
    target_type: Literal["section", "synthesis"] = Field(
        description="修改目标类型：section 重写指定章节，synthesis 修改摘要和结论。",
    )
    target_id: str = Field(
        min_length=1,
        description="section 类型为现有 section_id；synthesis 类型固定为 review。",
    )
    issue: str = Field(
        min_length=1,
        description="当前草稿中影响理解或结论准确性的具体写作问题。",
    )
    required_change: str = Field(
        min_length=1,
        description="节点必须执行的定向写作修改，不涉及新增证据、Claim 或 Framework。",
    )
    acceptance_criteria: str = Field(
        min_length=1,
        description="复查时用于判断该写作问题已解决的明确标准。",
    )


class WritingReviewResult(BaseModel):
    decision: Literal["pass", "revise", "blocked"] = Field(
        description="审核决定：pass 可交付，revise 执行修改，blocked 表示问题无法仅靠写作解决。",
    )
    summary: str = Field(
        min_length=1,
        description="概括审核结论、主要问题或受阻原因。",
    )
    revisions: list[WritingRevision] = Field(
        default_factory=list,
        description="必须执行的写作修改项；仅 decision 为 revise 时非空，pass 或 blocked 时为空。",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="不影响交付的一般润色建议，不触发下一轮修改。",
    )

    @model_validator(mode="after")
    def validate_decision(self):
        if (self.decision == "revise") != bool(self.revisions):
            raise ValueError("only revise may include required writing revisions")
        return self


class ReflectReviewNode:
    def __init__(self, *, artifact_store=None, model=None, chat: ChatClient | None = None):
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.model = model or (chat or ChatClient()).structured(
            WritingReviewResult, options=ModelOptions(temperature=0)
        )

    async def __call__(self, state):
        try:
            draft, corpus, claims_payload, verification_payload, framework_payload = (
                await asyncio.gather(
                    self.artifact_store.read_json_uri(state["review_draft_artifact_ref"]),
                    self.artifact_store.read_json_uri(state["corpus_artifact_ref"]),
                    self.artifact_store.read_json_uri(state["claims_artifact_ref"]),
                    self.artifact_store.read_json_uri(state["claim_verification_artifact_ref"]),
                    self.artifact_store.read_json_uri(state["framework_artifact_ref"]),
                )
            )
            previous_plan = (
                await self.artifact_store.read_json_uri(state["writing_review_plan_ref"])
                if state.get("writing_review_plan_ref")
                else None
            )
            framework = framework_payload["framework"]
            supported_claims = self._supported_claims(
                claims_payload["claims"], verification_payload["claims"]
            )
            hard_issues = self._check_hard_constraints(
                review_draft=draft,
                corpus_payload=corpus,
                framework=framework,
                require_citations=bool(supported_claims),
            )
            result = await invoke(
                self.model,
                WritingReviewResult,
                WRITING_REVIEW_PROMPT,
                {
                    "review_mode": "acceptance" if previous_plan else "initial",
                    "framework": {
                        "title": framework["title"],
                        "scope": framework["scope"],
                        "sections": [
                            {
                                key: section[key]
                                for key in (
                                    "section_id",
                                    "title",
                                    "description",
                                    "discussion_questions",
                                )
                            }
                            for section in framework["sections"]
                        ],
                    },
                    "abstract": draft["abstract"],
                    "body_markdown": draft["body_markdown"],
                    "conclusion": draft["conclusion"],
                    "supported_claims": supported_claims,
                    "previous_plan": previous_plan,
                    "hard_issues": hard_issues,
                },
            )
            self._validate_targets(result["revisions"], framework)
            return await self._complete_reflection(state, result, hard_issues)
        except Exception as exc:
            return failed(
                f"review reflection failed: {exc}",
                error_code="REFLECTION_FAILED",
                retryable=True,
            )

    @staticmethod
    def _supported_claims(claims, verification):
        claims_by_id = {claim["claim_id"]: claim for claim in claims}
        supported = []
        for verdict in verification:
            claim = claims_by_id.get(verdict["claim_id"])
            if (
                claim
                and not claim["withdrawn_reason"]
                and verdict["claim_hash"] == claim["claim_hash"]
                and verdict["status"] == "supported"
            ):
                supported.append(
                    {key: claim[key] for key in ("claim_id", "section_id", "text")}
                )
        return supported

    @staticmethod
    def _validate_targets(revisions, framework):
        section_ids = {section["section_id"] for section in framework["sections"]}
        for revision in revisions:
            if revision["target_type"] == "section":
                if revision["target_id"] not in section_ids:
                    raise ValueError("writing revision targets an unknown section")
            elif revision["target_id"] != "review":
                raise ValueError("synthesis revision must target review")

    async def _complete_reflection(self, state, result, hard_issues):
        decision = result["decision"]
        stop_reason = None
        if decision == "blocked":
            effective_decision = "blocked"
            stop_reason = "model_blocked"
        elif decision == "pass" and hard_issues:
            effective_decision = "blocked"
            stop_reason = "hard_constraints_failed"
        elif (
            decision == "revise"
            and state["writing_revision_round"] >= state["max_writing_revision_rounds"]
        ):
            effective_decision = "blocked"
            stop_reason = "writing_revision_limit_reached"
        else:
            effective_decision = decision

        report = await save(
            self.artifact_store,
            state,
            "reflection_report",
            {
                "review_draft_artifact_ref": state["review_draft_artifact_ref"],
                "writing_revision_round": state["writing_revision_round"],
                "previous_plan_ref": state.get("writing_review_plan_ref"),
                "model_result": result,
                "hard_issues": hard_issues,
                "effective_decision": effective_decision,
                "stop_reason": stop_reason,
            },
        )
        update = {"reflection_report_artifact_ref": report}
        if effective_decision == "pass":
            return {**update, "stage": "finalizing_handoff", "status": "running"}
        if effective_decision == "blocked":
            error_code = (
                "WRITING_REVISION_LIMIT_REACHED"
                if stop_reason == "writing_revision_limit_reached"
                else "WRITING_REVIEW_BLOCKED"
            )
            return failed(result["summary"], **update, error_code=error_code)

        next_round = state["writing_revision_round"] + 1
        plan = await save(
            self.artifact_store,
            state,
            "writing_review_plan",
            {
                "review_draft_artifact_ref": state["review_draft_artifact_ref"],
                "writing_revision_round": next_round,
                "revisions": result["revisions"],
            },
        )
        stage = (
            "rendering_sections"
            if any(item["target_type"] == "section" for item in result["revisions"])
            else "assembling_review"
        )
        return {
            **update,
            "writing_review_plan_ref": plan,
            "writing_revision_round": next_round,
            "revision_items": result["revisions"],
            "stage": stage,
            "status": "running",
        }

    def _check_hard_constraints(
        self,
        *,
        review_draft,
        corpus_payload,
        framework,
        require_citations=True,
    ):
        text = "\n".join(
            [
                review_draft.get("abstract", ""),
                review_draft.get("body_markdown", ""),
                review_draft.get("conclusion", ""),
            ]
        )
        anchors = citation_anchor_ids(text)
        known_paper_ids = {str(paper["paper_id"]) for paper in corpus_payload["papers"]}
        section_ids = {section["section_id"] for section in review_draft["sections"]}
        issues = []
        if not review_draft.get("body_markdown", "").strip():
            issues.append("review body is empty")
        if not review_draft.get("abstract", "").strip():
            issues.append("review abstract is empty")
        if not review_draft.get("conclusion", "").strip():
            issues.append("review conclusion is empty")
        if require_citations and not anchors:
            issues.append("review has no citation anchors")
        for section in framework["sections"]:
            if section["section_id"] not in section_ids:
                issues.append(f"review omits framework section: {section['section_id']}")
        for paper_id in sorted(
            unsupported_synthesis_anchor_ids(
                body_markdown=review_draft.get("body_markdown", ""),
                abstract=review_draft.get("abstract", ""),
                conclusion=review_draft.get("conclusion", ""),
            )
        ):
            issues.append(
                "synthesis cites a paper outside the evidence-backed body: "
                f"REF_{paper_id}"
            )
        for paper_id in anchors:
            if not paper_id.isdigit():
                issues.append(f"malformed citation anchor: REF_{paper_id}")
            elif paper_id not in known_paper_ids:
                issues.append(f"citation is outside task corpus: REF_{paper_id}")
        return list(dict.fromkeys(issues))
