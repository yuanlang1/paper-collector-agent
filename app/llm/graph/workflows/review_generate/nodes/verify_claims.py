from typing import Literal

from pydantic import BaseModel, Field

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.model_factory import create_validated_structured_chat_model
from app.llm.graph.workflows.review_generate.contracts import (
    RevisionItem,
    batches,
    content_hash,
    invoke,
    save,
    text_windows,
    verified_pack,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import failed
from app.llm.graph.workflows.review_generate.revisions import schedule_revision


EVIDENCE_SELECTION_PROMPT = (
    "从本批原文提取与论点相关的支持和反驳摘录，保留条件、数值和不可比因素。" "摘录必须逐字可定位；不得遗漏反证，不能用摘要代替摘录。仅作证据选择，不判断整个论点。"
)

CLAIM_VERIFICATION_PROMPT = (
    "核验当前论点措辞是否由原文支持。检查对象、条件、数值、因果、可比性及反证。"
    "相关或片段数量足够不等于支持；不得因缺失片段断言领域缺乏研究。"
    "摘录必须逐字来自输入，保留支持和反对依据。存在条件差异则收缩措辞或呈现分歧。"
    "非 supported 必须给出 claim 或 retrieval 修订项，target_id 使用当前 claim_id。"
    "supported 不给修订项；证据不可承载过宽论点时要求拆分或收缩。"
)


class EvidenceQuote(BaseModel):
    paper_id: str
    chunk_id: str
    quote: str = Field(min_length=1)
    relation: Literal["supports", "opposes"]


class Verdict(BaseModel):
    status: Literal["supported", "mixed", "contradicted", "insufficient"]
    evidence: list[EvidenceQuote]
    reason: str
    revisions: list[RevisionItem]


class EvidenceSelection(BaseModel):
    evidence: list[EvidenceQuote]
    limitations: str


class VerifyClaimsNode:
    def __init__(self, *, artifact_store=None, model=None, selection_model=None):
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.model = model or create_validated_structured_chat_model(Verdict, temperature=0)
        self.selection_model = selection_model

    async def _context(self, claim, evidence):
        windows = [{**item, **window} for item in evidence for window in text_windows(item["text"])]
        groups = list(batches(windows, budget=20000))
        if len(groups) <= 1:
            return evidence
        model = self.selection_model or create_validated_structured_chat_model(
            EvidenceSelection, temperature=0
        )
        selected = []
        for group in groups:
            result = await invoke(
                model,
                EvidenceSelection,
                EVIDENCE_SELECTION_PROMPT,
                {"claim": claim, "original_evidence": group},
            )
            for quote in result["evidence"]:
                if not any(
                    quote["paper_id"] == item["paper_id"]
                    and quote["chunk_id"] == item["chunk_id"]
                    and quote["quote"] in item["text"]
                    for item in group
                ):
                    raise ValueError("selected quote is not locatable in its original window")
                selected.append(
                    {**quote, "text": quote["quote"], "limitations": result["limitations"]}
                )
        if sum(len(str(item)) for item in selected) > 24000:
            return None
        return selected

    async def __call__(self, state):
        try:
            claims = (await self.artifact_store.read_json_uri(state["claims_artifact_ref"]))[
                "claims"
            ]
            ledger = {
                item["claim_id"]: item
                for item in (
                    await self.artifact_store.read_json_uri(state["evidence_ledger_artifact_ref"])
                )["claims"]
            }
            previous = {}
            if state.get("claim_verification_artifact_ref"):
                previous = {
                    item["claim_id"]: item
                    for item in (
                        await self.artifact_store.read_json_uri(
                            state["claim_verification_artifact_ref"]
                        )
                    )["claims"]
                }
            verdicts, revisions = [], []
            for claim in claims:
                if claim["withdrawn_reason"]:
                    continue
                if ledger[claim["claim_id"]]["claim_hash"] != claim["claim_hash"]:
                    raise ValueError("evidence ledger belongs to a previous claim version")
                evidence = ledger[claim["claim_id"]]["chunk_snippets"]
                digest = content_hash(evidence)
                cached = previous.get(claim["claim_id"])
                if (
                    cached
                    and cached["claim_hash"] == claim["claim_hash"]
                    and cached["evidence_digest"] == digest
                ):
                    verdict = cached
                else:
                    context = await self._context(claim, evidence)
                    if context is None:
                        verdict = {
                            "status": "insufficient",
                            "evidence": [],
                            "reason": "evidence budget exceeded",
                            "revisions": [
                                dict(
                                    target_type="claim",
                                    target_id=claim["claim_id"],
                                    issue="Evidence exceeds the input budget",
                                    required_change="Narrow the claim scope",
                                    acceptance_criteria=(
                                        "A complete evidence package fits the budget"
                                    ),
                                    queries=[],
                                )
                            ],
                        }
                    else:
                        verdict = await invoke(
                            self.model,
                            Verdict,
                            CLAIM_VERIFICATION_PROMPT,
                            {"claim": claim, "original_evidence": context},
                        )
                    sources = {(item["paper_id"], item["chunk_id"]): item for item in evidence}
                    for quote in verdict["evidence"]:
                        source = sources.get((quote["paper_id"], quote["chunk_id"]))
                        if source is None or quote["quote"] not in source["text"]:
                            raise ValueError(
                                "verification quote is not locatable in original evidence"
                            )
                        quote.update(
                            {
                                key: source.get(key)
                                for key in (
                                    "page_start",
                                    "page_end",
                                    "page_numbers",
                                    "section_path",
                                )
                            }
                        )
                        start = source["text"].index(quote["quote"])
                        quote["char_start"] = start
                        quote["char_end"] = start + len(quote["quote"])
                        quote["text"] = source["text"][
                            max(0, start - 500) : quote["char_end"] + 500
                        ]
                        quote["evidence_id"] = content_hash(
                            [quote["paper_id"], quote["chunk_id"], quote["quote"]]
                        )[:20]
                    verdict.update(
                        claim_id=claim["claim_id"],
                        claim_hash=claim["claim_hash"],
                        evidence_digest=digest,
                    )
                    if len(str({"claim": claim, "evidence": verdict["evidence"]})) > 24000:
                        verdict.update(
                            status="insufficient",
                            revisions=[
                                dict(
                                    target_type="claim",
                                    target_id=claim["claim_id"],
                                    issue="Final evidence package exceeds writing budget",
                                    required_change=(
                                        "Narrow the claim without dropping necessary sources"
                                    ),
                                    acceptance_criteria=(
                                        "Complete evidence package fits one writing call"
                                    ),
                                    queries=[],
                                )
                            ],
                        )
                    if verdict["status"] == "supported":
                        verified_pack(claim, verdict)
                        if verdict["revisions"]:
                            raise ValueError("supported verdict cannot request claim changes")
                    elif not verdict["revisions"]:
                        raise ValueError("unsupported verdict needs an executable revision")
                    if any(
                        item["target_type"] not in {"claim", "retrieval"}
                        or item["target_id"] != claim["claim_id"]
                        for item in verdict["revisions"]
                    ):
                        raise ValueError("verification revision must target the current claim")
                verdicts.append(verdict)
                revisions.extend(verdict["revisions"])
            supported = {item["claim_id"] for item in verdicts if item["status"] == "supported"}
            missing = [
                question
                for question in state["review_focus"]["research_questions"]
                if question["core"]
                and not any(
                    claim["question_id"] == question["question_id"]
                    and claim["claim_id"] in supported
                    for claim in claims
                )
            ]
            ref = await save(
                self.artifact_store,
                state,
                "claim_verification",
                {"claims": verdicts, "unanswered_core_questions": missing},
            )
            update = {"claim_verification_artifact_ref": ref}
            if revisions:
                revisions.extend(
                    item
                    for item in state.get("revision_items", [])
                    if item["target_type"] in {"framework", "section", "synthesis"}
                )
                section_ids = []
                if state.get("framework_artifact_ref"):
                    section_ids = [
                        section["section_id"]
                        for section in (
                            await self.artifact_store.read_json_uri(state["framework_artifact_ref"])
                        )["framework"]["sections"]
                    ]
                result = await schedule_revision(
                    self.artifact_store,
                    {**state, **update},
                    revisions,
                    claim_ids=[claim["claim_id"] for claim in claims],
                    section_ids=section_ids,
                )
                return {**update, **result}
            if missing or not supported:
                return failed(
                    "core research questions lack verified evidence",
                    **update,
                    error_code="EVIDENCE_INSUFFICIENT",
                )
            return {**update, "stage": "generating_framework", "status": "running"}
        except Exception as exc:
            return failed(
                f"claim verification failed: {exc}",
                error_code="VERIFICATION_FAILED",
                retryable=True,
            )
