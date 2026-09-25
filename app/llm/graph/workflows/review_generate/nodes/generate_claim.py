from pydantic import BaseModel, Field

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.contracts import (
    CandidateClaim,
    batches,
    claim_hash,
    invoke,
    revisions_for,
    save,
    validate_claims,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import failed
from app.llm.provider import ChatClient, ModelOptions


CLAIM_GENERATION_PROMPT = (
    "围绕当前章节的讨论问题比较论文，生成具体、可核验的候选综合论点，不按论文罗列。"
    "论文画像不是原文证据，不预设一致性或优越性；不可新增论文。"
    "跨论文比较必须要求 multiple_fulltext。保留分歧、条件和局限；检索查询应同时寻找支持、相反结果和适用条件；使用输出语言。"
)

CLAIM_MERGE_PROMPT = (
    "跨批次综合已有候选论点，合并重复，保留一致、差异与条件。"
    "不得引入输入之外的事实；所有结论仍须原文核验。"
)

CLAIM_REVISION_PROMPT = (
    "根据原文和修订要求调整当前论点。收缩范围或明确分歧；条件、对象或结论不可兼容时拆分为多个独立论点。"
    "不能成立的非核心候选可填写 withdrawn_reason。不可换章节，不得虚构。每条论点可改写查询以寻找支持或反证。"
)


class ClaimsPlan(BaseModel):
    claims: list[CandidateClaim]
    unanswered_reason: str = ""


class ClaimRevisionPlan(BaseModel):
    claims: list[CandidateClaim] = Field(min_length=1)


class GenerateClaimsNode:
    def __init__(self, *, artifact_store=None, model=None, revision_model=None, chat: ChatClient | None = None):
        self.artifact_store = artifact_store or LocalArtifactStore()
        client = chat or ChatClient()
        self.model = model or client.structured(ClaimsPlan, options=ModelOptions(temperature=0))
        self.revision_model = revision_model or client.structured(
            ClaimRevisionPlan, options=ModelOptions(temperature=0)
        )

    async def _generate_for_section(self, state, section, studies, existing, additions):
        relevant = [
            study for study in studies if study["paper_id"] in section["relevant_paper_ids"]
        ]
        candidates = []
        reason = ""
        groups = list(batches(relevant))
        for group in groups:
            result = await invoke(
                self.model,
                ClaimsPlan,
                CLAIM_GENERATION_PROMPT,
                {
                    "section": section,
                    "studies": group,
                    "existing_candidates": [claim["text"] for claim in existing + candidates],
                    "revisions": additions,
                    "output_language": state["language"],
                },
            )
            candidates.extend(result["claims"])
            if result["unanswered_reason"]:
                reason = result["unanswered_reason"]
        if len(groups) > 1 and candidates:
            result = await invoke(
                self.model,
                ClaimsPlan,
                CLAIM_MERGE_PROMPT,
                {
                    "section": section,
                    "candidates": candidates,
                    "output_language": state["language"],
                },
            )
            candidates = result["claims"]
            reason = result["unanswered_reason"]
        return candidates, reason

    async def __call__(self, state):
        try:
            framework_payload = await self.artifact_store.read_json_uri(
                state["framework_artifact_ref"]
            )
            framework = framework_payload["framework"]
            studies = (
                await self.artifact_store.read_json_uri(state["study_records_artifact_ref"])
            )["studies"]
            payload = (
                await self.artifact_store.read_json_uri(state["claims_artifact_ref"])
                if state.get("claims_artifact_ref")
                else {"claims": [], "unanswered": []}
            )
            section_ids = {section["section_id"] for section in framework["sections"]}
            changed = set(state.get("changed_section_ids", []))
            claims = [
                claim
                for claim in payload["claims"]
                if claim["section_id"] in section_ids and claim["section_id"] not in changed
            ]
            unanswered = [
                item
                for item in payload["unanswered"]
                if item["section_id"] in section_ids and item["section_id"] not in changed
            ]
            claim_revisions = revisions_for(state, "claim")
            targets = {item["target_id"] for item in claim_revisions}
            if not targets <= {claim["claim_id"] for claim in claims}:
                raise ValueError("unknown claim revision target")

            revised = []
            for claim in claims:
                if claim["claim_id"] not in targets:
                    revised.append(claim)
                    continue
                evidence = await self.artifact_store.read_json_uri(state["evidence_ledger_artifact_ref"])
                result = await invoke(
                    self.revision_model,
                    ClaimRevisionPlan,
                    CLAIM_REVISION_PROMPT,
                    {
                        "old_claim": claim,
                        "output_language": state["language"],
                        "revisions": revisions_for(state, "claim", claim["claim_id"]),
                        "evidence": next(
                            item for item in evidence["claims"] if item["claim_id"] == claim["claim_id"]
                        ),
                    },
                )
                replacements = result["claims"]
                ids = (
                    [claim["claim_id"]]
                    if len(replacements) == 1
                    else [f'{claim["claim_id"]}_{index}' for index in range(1, len(replacements) + 1)]
                )
                for replacement, claim_id in zip(replacements, ids):
                    replacement.update(claim_id=claim_id, section_id=claim["section_id"])
                    replacement["claim_hash"] = claim_hash(replacement)
                revised.extend(replacements)
            claims = revised

            additions = {
                section["section_id"]: revisions_for(state, "add_claim", section["section_id"])
                for section in framework["sections"]
            }
            sections_to_generate = changed | {
                section_id for section_id, items in additions.items() if items
            }
            if not payload["claims"]:
                sections_to_generate = section_ids
            used_ids = {claim["claim_id"] for claim in claims}
            next_index = len(claims) + 1
            for section in framework["sections"]:
                if section["section_id"] not in sections_to_generate:
                    continue
                candidates, reason = await self._generate_for_section(
                    state,
                    section,
                    studies,
                    [claim for claim in claims if claim["section_id"] == section["section_id"]],
                    additions[section["section_id"]],
                )
                unanswered = [
                    item for item in unanswered if item["section_id"] != section["section_id"]
                ]
                if reason:
                    unanswered.append({"section_id": section["section_id"], "reason": reason})
                for claim in candidates:
                    while f"claim_{next_index:03d}" in used_ids:
                        next_index += 1
                    claim_id = f"claim_{next_index:03d}"
                    next_index += 1
                    used_ids.add(claim_id)
                    claim.update(claim_id=claim_id, section_id=section["section_id"])
                    claim["claim_hash"] = claim_hash(claim)
                    claims.append(claim)

            validate_claims(claims, framework, state["paper_ids_snapshot"])
            ref = await save(
                self.artifact_store, state, "claims", {"claims": claims, "unanswered": unanswered}
            )
            return {
                "claims_artifact_ref": ref,
                "changed_section_ids": [],
                "revision_items": [
                    item
                    for item in state.get("revision_items", [])
                    if item["target_type"] not in {"claim", "add_claim"}
                ],
                "stage": "retrieving_evidence",
                "status": "running",
            }
        except Exception as exc:
            return failed(
                f"claim generation failed: {exc}", error_code="CLAIMS_FAILED", retryable=True
            )
