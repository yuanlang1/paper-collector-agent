import json
from typing import Any

from pydantic import BaseModel, Field

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.citations import (
    citation_anchor_ids,
    unsupported_synthesis_anchor_ids,
)
from app.llm.graph.workflows.review_generate.contracts import (
    RevisionItem,
    batches,
    invoke,
    save,
    text_windows,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import failed
from app.llm.graph.workflows.review_generate.revisions import schedule_revision
from app.llm.provider import ChatClient, ModelOptions


CLAIM_REFLECTION_PROMPT = (
    "检查论点在本章节中的真实论证。arguments 是正文实际使用的摘录；verification.evidence 是该论点的核验证据片段。"
    "当 evidence_fragment 为 true 时，只判断该片段和正文片段的忠实性，不重新判断整体来源数量。"
    "核对引用绑定、适用范围、数值、比较条件、反证和扩大论断。发现问题时提出 claim、retrieval 或 section 修订。"
)

SECTION_REFLECTION_PROMPT = (
    "检查本章节是否回答讨论问题，是否遗漏画像中已经呈现的重要发现、分歧或局限，并检查结构、重复和衔接。"
    "画像只能指出需要补查的方向，不能直接当作正文证据。遗漏论点使用 add_claim，target_id 使用 section_id。"
    "章节文本问题使用 section，结构问题使用 framework。"
)

REVIEW_REFLECTION_PROMPT = (
    "审核真实 body_markdown 中的新增事实、章节覆盖、重复和矛盾，并检查摘要结论是否扩大正文结论。"
    "正文为分批输入，不因其他批次内容缺席判断遗漏。摘要结论修改目标为 synthesis/review；结构修改为 framework/review。"
    "硬检查问题必须解决。不允许虚构系统综述执行过程。"
)


class ReflectionIssue(BaseModel):
    severity: str
    category: str
    description: str


class ReflectionResult(BaseModel):
    satisfied: bool
    summary: str = ""
    issues: list[ReflectionIssue] = Field(default_factory=list)
    revisions: list[RevisionItem] = Field(default_factory=list)


class ReflectReviewNode:
    def __init__(self, *, artifact_store=None, model=None, chat: ChatClient | None = None):
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.model = model or (chat or ChatClient()).structured(
            ReflectionResult, options=ModelOptions(temperature=0)
        )

    def _claim_checks(self, claim, verification, arguments):
        base = {
            "claim": claim,
            "verification": {
                "status": verification["status"],
                "reason": verification["reason"],
            },
        }
        argument_windows = [
            {**argument, "text": window["text"], "char_start": window["char_start"]}
            for argument in arguments
            for window in text_windows(argument["text"])
        ] or [{}]
        evidence_fragments = []
        for evidence in verification["evidence"]:
            if len(json.dumps(evidence, ensure_ascii=False)) <= 6000:
                evidence_fragments.append(evidence)
                continue
            evidence_fragments.extend(
                {
                    key: evidence[key]
                    for key in ("evidence_id", "paper_id", "chunk_id", "relation")
                }
                | {"text": window["text"]}
                for window in text_windows(evidence["text"], size=5000, overlap=0)
            )
        checks = []
        for argument_group in batches(argument_windows, budget=8000):
            payload = {**base, "arguments": argument_group}
            evidence_budget = 24000 - len(json.dumps(payload, ensure_ascii=False))
            for evidence_group in batches(evidence_fragments, budget=evidence_budget):
                checks.append(
                    {
                        **payload,
                        "verification": {**payload["verification"], "evidence": evidence_group},
                        "evidence_fragment": len(evidence_fragments) != len(evidence_group),
                    }
                )
        return checks

    async def __call__(self, state):
        try:
            draft = await self.artifact_store.read_json_uri(state["review_draft_artifact_ref"])
            corpus = await self.artifact_store.read_json_uri(state["corpus_artifact_ref"])
            studies = (
                await self.artifact_store.read_json_uri(state["study_records_artifact_ref"])
            )["studies"]
            claims = (await self.artifact_store.read_json_uri(state["claims_artifact_ref"]))[
                "claims"
            ]
            verification = (
                await self.artifact_store.read_json_uri(state["claim_verification_artifact_ref"])
            )["claims"]
            framework = (await self.artifact_store.read_json_uri(state["framework_artifact_ref"]))[
                "framework"
            ]
            claims_by_id = {claim["claim_id"]: claim for claim in claims}
            verification_by_id = {item["claim_id"]: item for item in verification}
            reports = []
            for section in draft["sections"]:
                for claim_id in section["used_claim_ids"]:
                    reports.extend(
                        await self._reflect_claim_checks(
                            section["section_id"],
                            self._claim_checks(
                                claims_by_id[claim_id],
                                verification_by_id[claim_id],
                                [
                                    argument
                                    for argument in section["arguments"]
                                    if claim_id in argument["claim_ids"]
                                ],
                            ),
                            state,
                        )
                    )
                framework_section = next(
                    item for item in framework["sections"] if item["section_id"] == section["section_id"]
                )
                profiles = [
                    study
                    for study in studies
                    if study["paper_id"] in framework_section["relevant_paper_ids"]
                ]
                section_claims = [
                    claim for claim in claims if claim["section_id"] == section["section_id"]
                ]
                for window in text_windows(section["text"]):
                    reports.append(
                        await invoke(
                            self.model,
                            ReflectionResult,
                            SECTION_REFLECTION_PROMPT,
                            {
                                "section": framework_section,
                                "actual_text": window,
                                "section_summary": section["summary"],
                                "study_profiles": profiles,
                                "section_claims": section_claims,
                                "previous_revisions": state.get("revision_items", []),
                            },
                        )
                    )
            hard_issues = self._check_hard_constraints(
                review_draft=draft,
                corpus_payload=corpus,
                require_citations=any(item["status"] == "supported" for item in verification),
            )
            for group in batches(list(text_windows(draft["body_markdown"])), budget=24000):
                reports.append(
                    await invoke(
                        self.model,
                        ReflectionResult,
                        REVIEW_REFLECTION_PROMPT,
                        {
                            "framework": framework,
                            "body_sections": group,
                            "verified_claims": [
                                claim for claim in claims if not claim["withdrawn_reason"]
                            ],
                            "abstract": draft["abstract"],
                            "conclusion": draft["conclusion"],
                            "system_hard_issues": hard_issues,
                            "previous_revisions": state.get("revision_items", []),
                        },
                    )
                )
            satisfied = not hard_issues and all(
                report["satisfied"]
                and not any(
                    issue["severity"] in {"critical", "major"} for issue in report["issues"]
                )
                for report in reports
            )
            items = [item for report in reports for item in report["revisions"]]
            ref = await save(
                self.artifact_store,
                state,
                "reflection_report",
                {"satisfied": satisfied, "hard_issues": hard_issues, "reports": reports},
            )
            update = {"reflection_report_artifact_ref": ref}
            if satisfied and not items:
                return {**update, "stage": "finalizing_handoff", "status": "running"}
            result = await schedule_revision(
                self.artifact_store,
                {**state, **update},
                items,
                claim_ids=[claim["claim_id"] for claim in claims],
                section_ids=[section["section_id"] for section in framework["sections"]],
            )
            return {**update, **result}
        except Exception as exc:
            return failed(
                f"review reflection failed: {exc}", error_code="REFLECTION_FAILED", retryable=True
            )

    async def _reflect_claim_checks(self, section_id, checks, state):
        return [
            await invoke(
                self.model,
                ReflectionResult,
                CLAIM_REFLECTION_PROMPT,
                {
                    "section_id": section_id,
                    "claim_checks": [check],
                    "previous_revisions": state.get("revision_items", []),
                },
            )
            for check in checks
        ]

    def _check_hard_constraints(
        self,
        *,
        review_draft: dict[str, Any],
        corpus_payload: dict[str, Any],
        require_citations: bool = True,
    ) -> list[str]:
        text = "\n".join(
            [
                review_draft.get("abstract", ""),
                review_draft.get("body_markdown", ""),
                review_draft.get("conclusion", ""),
            ]
        )
        anchors = citation_anchor_ids(text)
        known_paper_ids = {str(paper["paper_id"]) for paper in corpus_payload["papers"]}
        issues = []
        if not review_draft.get("body_markdown", "").strip():
            issues.append("review body is empty")
        if not review_draft.get("abstract", "").strip():
            issues.append("review abstract is empty")
        if not review_draft.get("conclusion", "").strip():
            issues.append("review conclusion is empty")
        if require_citations and not anchors:
            issues.append("review has no citation anchors")
        for paper_id in sorted(
            unsupported_synthesis_anchor_ids(
                body_markdown=review_draft.get("body_markdown", ""),
                abstract=review_draft.get("abstract", ""),
                conclusion=review_draft.get("conclusion", ""),
            )
        ):
            issues.append(
                "synthesis cites a paper outside the evidence-backed body: " f"REF_{paper_id}"
            )
        for paper_id in anchors:
            if not paper_id.isdigit():
                issues.append(f"malformed citation anchor: REF_{paper_id}")
            elif paper_id not in known_paper_ids:
                issues.append(f"citation is outside task corpus: REF_{paper_id}")
        return list(dict.fromkeys(issues))
