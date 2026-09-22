from typing import Any

from pydantic import BaseModel, Field

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.model_factory import create_validated_structured_chat_model
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


SECTION_REFLECTION_PROMPT = (
    "检查真实正文中每个事实，不能只信 used_claim_ids。核对原文适用范围、引用绑定、"
    "数值、比较条件、反证和新增扩大论断。claims 是此章节完整的已核验论点列表，"
    "超出该列表的具体事实必须修订或重新核验。本次只看到部分证据窗口，"
    "不能仅因本窗口缺证据判为无依据；需要完整核验时提出 claim 或 retrieval 修订。"
    "修订明确问题、修改要求与验收条件，章节修改使用当前 section_id。"
)

REVIEW_REFLECTION_PROMPT = (
    "审核真实 body_markdown 中的新增事实、研究问题覆盖、章节结构、重复和矛盾，"
    "与完整已核验论点列表比较，检查摘要结论是否扩大正文结论。"
    "正文是分批输入，不因其他批次内容缺席判定遗漏。摘要结论修改目标为 synthesis/review；"
    "结构修改为 framework/review。硬检查问题必须解决。不允许虚构系统综述执行过程。"
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
    def __init__(self, *, artifact_store=None, model=None):
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.model = model or create_validated_structured_chat_model(
            ReflectionResult, temperature=0
        )

    async def __call__(self, state):
        try:
            draft = await self.artifact_store.read_json_uri(state["review_draft_artifact_ref"])
            corpus = await self.artifact_store.read_json_uri(state["corpus_artifact_ref"])
            claims = (await self.artifact_store.read_json_uri(state["claims_artifact_ref"]))[
                "claims"
            ]
            verification = (
                await self.artifact_store.read_json_uri(state["claim_verification_artifact_ref"])
            )["claims"]
            ledger = {
                item["claim_id"]: item
                for item in (
                    await self.artifact_store.read_json_uri(state["evidence_ledger_artifact_ref"])
                )["claims"]
            }
            framework = (await self.artifact_store.read_json_uri(state["framework_artifact_ref"]))[
                "framework"
            ]
            reports = []
            for section in draft["sections"]:
                section_claims = [
                    claim for claim in claims if claim["claim_id"] in section["used_claim_ids"]
                ]
                checks = [
                    item for item in verification if item["claim_id"] in section["used_claim_ids"]
                ]
                # Bound raw-text windows and overlap them for boundary-spanning issues.
                records = []
                for check in checks:
                    for evidence in ledger[check["claim_id"]]["chunk_snippets"]:
                        for window in text_windows(evidence["text"]):
                            records.append(
                                {
                                    "claim_id": check["claim_id"],
                                    "verdict": check["status"],
                                    **evidence,
                                    **window,
                                }
                            )
                for window in text_windows(section["text"]):
                    for group in list(batches(records, budget=12000)) or [[]]:
                        reports.append(
                            await invoke(
                                self.model,
                                ReflectionResult,
                                SECTION_REFLECTION_PROMPT,
                                {
                                    "section_id": section["section_id"],
                                    "actual_text": window,
                                    "claims": section_claims,
                                    "original_evidence": group,
                                    "previous_revisions": state.get("revision_items", []),
                                },
                            )
                        )
            hard_issues = self._check_hard_constraints(review_draft=draft, corpus_payload=corpus)
            for group in batches(list(text_windows(draft["body_markdown"])), budget=24000):
                reports.append(
                    await invoke(
                        self.model,
                        ReflectionResult,
                        REVIEW_REFLECTION_PROMPT,
                        {
                            "focus": state["review_focus"],
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

    def _check_hard_constraints(
        self, *, review_draft: dict[str, Any], corpus_payload: dict[str, Any],
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

        if not anchors:
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
