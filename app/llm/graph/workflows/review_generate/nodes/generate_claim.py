from pydantic import BaseModel, Field

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.provider import ChatClient, ModelOptions
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


CLAIM_REVISION_PROMPT = (
    "根据原文和修订要求调整当前论点。收缩范围或明确分歧；条件、对象或结论不可兼容时拆分为多个独立论点。"
    "不能成立的非核心候选可填写 withdrawn_reason。不可换题，不得虚构。每条论点可改写查询以寻找支持或反证。"
)

CLAIM_GENERATION_PROMPT = (
    "围绕研究问题比较论文，生成具体、可核验的候选综合论点，不按论文罗列。"
    "概览不等于原文证据，不预设一致性或优越性；不可新增论文。"
    "跨论文比较必须要求 multiple_fulltext。无法提出论点时说明原因。使用输出语言。"
)

CLAIM_MERGE_PROMPT = "跨批次综合已有候选论点，合并重复，比较一致、差异与条件。" "保留不同结果和候选论文来源，不引入输入之外的事实；所有结论仍须原文核验。"


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

    async def __call__(self, state):
        try:
            focus = state["review_focus"]
            studies = (
                await self.artifact_store.read_json_uri(state["study_records_artifact_ref"])
            )["studies"]
            items = revisions_for(state, "claim")
            unanswered = []

            if state.get("claims_artifact_ref"):
                payload = await self.artifact_store.read_json_uri(state["claims_artifact_ref"])
                claims, unanswered = payload["claims"], payload["unanswered"]
                targets = {item["target_id"] for item in items}
                if not targets <= {claim["claim_id"] for claim in claims}:
                    raise ValueError("unknown claim revision target")
                evidence = await self.artifact_store.read_json_uri(
                    state["evidence_ledger_artifact_ref"]
                )
                updated_claims = []
                for claim in claims:
                    if claim["claim_id"] not in targets:
                        updated_claims.append(claim)
                        continue
                    result = await invoke(
                        self.revision_model,
                        ClaimRevisionPlan,
                        CLAIM_REVISION_PROMPT,
                        {
                            "old_claim": claim,
                            "output_language": state["language"],
                            "revisions": revisions_for(state, "claim", claim["claim_id"]),
                            "evidence": next(
                                item
                                for item in evidence["claims"]
                                if item["claim_id"] == claim["claim_id"]
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
                        replacement.update(claim_id=claim_id, question_id=claim["question_id"])
                        replacement["claim_hash"] = claim_hash(replacement)
                    updated_claims.extend(replacements)
                claims = updated_claims
            else:
                claims = []
                for question in focus["research_questions"]:
                    relevant = [
                        record
                        for record in studies
                        if record["paper_id"] in question["relevant_paper_ids"]
                    ]
                    question_claims = []
                    groups = list(batches(relevant))
                    for group in groups:
                        result = await invoke(
                            self.model,
                            ClaimsPlan,
                            CLAIM_GENERATION_PROMPT,
                            {
                                "question": question,
                                "studies": group,
                                "output_language": state["language"],
                                "existing_candidates": [claim["text"] for claim in question_claims],
                            },
                        )
                        question_claims.extend(result["claims"])
                        if result["unanswered_reason"]:
                            unanswered.append(
                                dict(
                                    question_id=question["question_id"],
                                    reason=result["unanswered_reason"],
                                )
                            )
                    if len(groups) > 1 and question_claims:
                        merged = await invoke(
                            self.model,
                            ClaimsPlan,
                            CLAIM_MERGE_PROMPT,
                            {
                                "question": question,
                                "candidates": question_claims,
                                "output_language": state["language"],
                            },
                        )
                        question_claims = merged["claims"]
                        if merged["unanswered_reason"]:
                            unanswered.append(
                                dict(
                                    question_id=question["question_id"],
                                    reason=merged["unanswered_reason"],
                                )
                            )
                    for claim in question_claims:
                        claim.update(
                            claim_id=f"claim_{len(claims)+1:03d}",
                            question_id=question["question_id"],
                        )
                        claim["claim_hash"] = claim_hash(claim)
                        claims.append(claim)
            validate_claims(claims, focus, state["paper_ids_snapshot"])
            ref = await save(
                self.artifact_store, state, "claims", {"claims": claims, "unanswered": unanswered}
            )
            return {"claims_artifact_ref": ref, "stage": "retrieving_evidence", "status": "running"}
        except Exception as exc:
            return failed(
                f"claim generation failed: {exc}", error_code="CLAIMS_FAILED", retryable=True
            )
