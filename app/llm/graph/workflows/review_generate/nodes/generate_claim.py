import asyncio
import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.contracts import (
    CandidateClaim,
    claim_hash,
    revisions_for,
    save,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import failed
from app.llm.provider import ChatClient, ModelOptions


CLAIM_GENERATION_PROMPT = """
    你是一名学术综述论点设计者。围绕当前 section 的 discussion_questions 比较和综合 studies，
    生成具体、可核验的候选论点，不按论文逐篇罗列。每个论点应服务于至少一个讨论问题，
    且不得与 existing_candidates 重复。论文画像不是原文证据，不得预设一致性、优越性或因果结论，
    也不可新增论文。

    保留分歧、条件和局限。跨论文比较必须使用 `comparative + multiple_fulltext`；
    明确归属于单篇研究的实验比较只能描述该研究结果，使用 `descriptive + fulltext`。
    每条论点只表达一个独立的核心判断；将同一发现的条件、局限和轻微改写合并到原论点，
    不按论文数量或讨论问题数量机械增加论点。
    本节最多生成 claim_budget 条论点，允许少于额度；按对讨论问题的重要性排序，不为填满额度生成内容。
    retrieval_queries 默认只用一条；只有独立的比较、条件或反证检索确有必要时才增加第二条。
    revisions 非空时，只执行其中针对当前章节的新增要求。使用 output_language。

    以下仅为格式示例；使用实际论文 ID 和章节内容，勿照抄：
    ```json
    {
    "claims": [
        {
        "text": "在给定研究条件下，两种方法的结果差异可能与测量设置有关。",
        "candidate_paper_ids": ["paper_001", "paper_002"],
        "retrieval_queries": ["方法 A 与方法 B 的结果比较及其测量条件"],
        "claim_type": "comparative",
        "evidence_requirement": "multiple_fulltext",
        "withdrawn_reason": "",
        "section_id": ""
        }
    ],
    "unanswered_reason": ""
    }
    ```
    无法基于论文画像形成候选时，返回：
    ```json
    {"claims": [], "unanswered_reason": "说明缺少哪类信息"}
    ```
""".strip()

CLAIM_REVISION_PROMPT = """
    你是一名学术综述论点修订者。根据 old_claim 的原文 evidence 和 revisions 调整当前论点，
    必须落实 required_change 和 acceptance_criteria。收缩范围、补充条件或降低强度；
    不能成立的非核心候选填写 withdrawn_reason。
    不可改变 section，不得虚构；ID、section_id 和哈希由程序处理。

    修订时必须同步检查正文、claim_type 和 evidence_requirement：跨论文比较仍是
    `comparative + multiple_fulltext`，不能只降低 evidence_requirement 而保留跨研究结论。
    若证据只描述单篇研究结果，改为 `descriptive + fulltext`；无法收缩为该类表述时撤回。
    只能返回一条替换论点或一条撤回记录。优先收缩为一条可由现有证据支持的论点，
    不得将限定条件、反例或措辞变化拆为新的论点。

    retrieval_queries 默认只用一条；只有独立的比较、条件或反证检索确有必要时才增加第二条。
    以下仅为格式示例，使用实际输入内容：
    替换：
    ```json
    {
    "claims": [{
        "text": "在特定条件下观察到相关性。",
        "candidate_paper_ids": ["paper_001"],
        "retrieval_queries": ["特定条件下的相关性"],
        "claim_type": "descriptive",
        "evidence_requirement": "fulltext",
        "withdrawn_reason": "",
        "section_id": ""
    }]
    }
    ```
    撤回：
    ```json
    {
    "claims": [{
        "text": "当前证据不足以支持原比较结论。",
        "candidate_paper_ids": ["paper_001"],
        "retrieval_queries": ["原比较结论的直接证据"],
        "claim_type": "critical",
        "evidence_requirement": "fulltext",
        "withdrawn_reason": "现有原文无法支持该候选论点。",
        "section_id": ""
    }]
    }
    ```
""".strip()


class ClaimsPlan(BaseModel):
    claims: list[CandidateClaim] = Field(
        description="围绕当前章节问题生成、尚待原文证据核验的候选论点列表。",
    )
    unanswered_reason: str = Field(
        default="",
        description="无法形成候选论点时说明缺失信息；有候选论点时通常为空。",
    )


class ClaimRevisionPlan(BaseModel):
    claims: list[CandidateClaim] = Field(
        min_length=1,
        max_length=1,
        description="替换原论点的一条候选论点；撤回也用一条带 withdrawn_reason 的论点表达。",
    )


MAX_CLAIMS_PER_SECTION = 4
MAX_CLAIMS_PER_REVIEW = 24


def claim_budgets(sections, claims):
    active = {
        section["section_id"]: sum(
            claim["section_id"] == section["section_id"]
            and not claim["withdrawn_reason"]
            for claim in claims
        )
        for section in sections
    }
    remaining = max(0, MAX_CLAIMS_PER_REVIEW - sum(active.values()))
    budgets = {section["section_id"]: 0 for section in sections}
    while remaining:
        allocated = False
        for section in sections:
            section_id = section["section_id"]
            if active[section_id] + budgets[section_id] >= MAX_CLAIMS_PER_SECTION:
                continue
            budgets[section_id] += 1
            remaining -= 1
            allocated = True
            if not remaining:
                break
        if not allocated:
            break
    return budgets


class GenerateClaimsNode:
    def __init__(
        self,
        *,
        artifact_store=None,
        model=None,
        revision_model=None,
        chat: ChatClient | None = None,
        max_concurrency: int = 4,
    ):
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.max_concurrency = max_concurrency
        client = chat or ChatClient()
        self.model = model or client.structured(ClaimsPlan, options=ModelOptions(temperature=0))
        self.revision_model = revision_model or client.structured(
            ClaimRevisionPlan, options=ModelOptions(temperature=0)
        )

    async def _generate_for_section(
        self, state, section, studies, existing, additions, claim_budget=MAX_CLAIMS_PER_SECTION
    ):
        relevant = [
            study for study in studies if study["paper_id"] in section["relevant_paper_ids"]
        ]
        result = (await self.model.ainvoke([
            SystemMessage(content=CLAIM_GENERATION_PROMPT + UNTRUSTED_INPUT_GUARD),
            HumanMessage(content=json.dumps({
                "section": section,
                "studies": relevant,
                "existing_candidates": [claim["text"] for claim in existing],
                "revisions": additions,
                "claim_budget": claim_budget,
                "output_language": state["language"],
            }, ensure_ascii=False)),
        ])).model_dump(mode="json")
        return result["claims"], result["unanswered_reason"]

    async def __call__(self, state):
        try:
            framework_payload = await self.artifact_store.read_json_uri(
                state["framework_artifact_ref"]
            )
            framework = framework_payload["framework"]
            sections = {section["section_id"]: section for section in framework["sections"]}
            studies = (
                await self.artifact_store.read_json_uri(state["study_records_artifact_ref"])
            )["studies"]
            payload = (
                await self.artifact_store.read_json_uri(state["claims_artifact_ref"])
                if state.get("claims_artifact_ref")
                else {"claims": [], "unanswered": []}
            )
            section_ids = set(sections)
            claims = [
                claim
                for claim in payload["claims"]
                if claim["section_id"] in section_ids
            ]
            unanswered = [
                item
                for item in payload["unanswered"]
                if item["section_id"] in section_ids
            ]
            claim_revisions = revisions_for(state, "claim")
            targets = {item["target_id"] for item in claim_revisions}

            async def wait_for(tasks):
                try:
                    return await asyncio.gather(*tasks)
                except (asyncio.CancelledError, Exception):
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise

            replacements_by_claim = {}
            if targets:
                evidence_by_claim = {
                    item["claim_id"]: item
                    for item in (
                        await self.artifact_store.read_json_uri(
                            state["evidence_ledger_artifact_ref"]
                        )
                    )["claims"]
                }
                semaphore = asyncio.Semaphore(self.max_concurrency)

                async def revise(claim):
                    async with semaphore:
                        result = (await self.revision_model.ainvoke([
                            SystemMessage(content=CLAIM_REVISION_PROMPT + UNTRUSTED_INPUT_GUARD),
                            HumanMessage(content=json.dumps({
                                "old_claim": claim,
                                "section": sections[claim["section_id"]],
                                "output_language": state["language"],
                                "revisions": revisions_for(
                                    state, "claim", claim["claim_id"]
                                ),
                                "evidence": evidence_by_claim[claim["claim_id"]],
                            }, ensure_ascii=False)),
                        ])).model_dump(mode="json")
                    return claim["claim_id"], result["claims"]

                replacements_by_claim = dict(
                    await wait_for(
                        [
                            asyncio.create_task(revise(claim))
                            for claim in claims
                            if claim["claim_id"] in targets
                        ]
                    )
                )

            revised = []
            for claim in claims:
                replacements = replacements_by_claim.get(claim["claim_id"])
                if replacements is None:
                    revised.append(claim)
                    continue
                replacement = replacements[0]
                replacement.update(
                    claim_id=claim["claim_id"], section_id=claim["section_id"]
                )
                replacement["claim_hash"] = claim_hash(replacement)
                revised.append(replacement)
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
            budgets = claim_budgets(framework["sections"], claims)
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def generate(section):
                async with semaphore:
                    candidates, reason = await self._generate_for_section(
                        state,
                        section,
                        studies,
                        [
                            claim
                            for claim in claims
                            if claim["section_id"] == section["section_id"]
                        ],
                        additions[section["section_id"]],
                        budgets[section["section_id"]],
                    )
                return section["section_id"], candidates, reason

            generated = {
                section["section_id"]: ([], "Claim budget exhausted for this section.")
                for section in framework["sections"]
                if section["section_id"] in sections_to_generate
                and not budgets[section["section_id"]]
            }
            generated.update(dict(
                (
                    section_id,
                    (candidates, reason),
                )
                for section_id, candidates, reason in await wait_for(
                    [
                        asyncio.create_task(generate(section))
                        for section in framework["sections"]
                        if section["section_id"] in sections_to_generate
                        and budgets[section["section_id"]]
                    ]
                )
            ))
            next_index = max(
                (int(claim["claim_id"].removeprefix("claim_")) for claim in claims),
                default=0,
            ) + 1
            warnings = list(state["warnings"])
            for section in framework["sections"]:
                if section["section_id"] not in sections_to_generate:
                    continue
                candidates, reason = generated[section["section_id"]]
                budget = budgets[section["section_id"]]
                if len(candidates) > budget:
                    warnings.append(
                        f"Claim budget limited section {section['section_id']} to {budget} claims."
                    )
                    candidates = candidates[:budget]
                unanswered = [
                    item for item in unanswered if item["section_id"] != section["section_id"]
                ]
                if reason:
                    unanswered.append({"section_id": section["section_id"], "reason": reason})
                for claim in candidates:
                    claim_id = f"claim_{next_index:03d}"
                    next_index += 1
                    claim.update(claim_id=claim_id, section_id=section["section_id"])
                    claim["claim_hash"] = claim_hash(claim)
                    claims.append(claim)

            paper_ids = set(state["paper_ids_snapshot"])
            if any(not set(claim["candidate_paper_ids"]) <= paper_ids for claim in claims):
                raise ValueError("claim references papers outside task")
            ref = await save(
                self.artifact_store, state, "claims", {"claims": claims, "unanswered": unanswered}
            )
            return {
                "claims_artifact_ref": ref,
                "revision_items": [
                    item
                    for item in state.get("revision_items", [])
                    if item["target_type"] not in {"claim", "add_claim"}
                ],
                "warnings": warnings,
                "stage": "retrieving_evidence",
                "status": "running",
            }
        except Exception as exc:
            return failed(
                f"claim generation failed: {exc}", error_code="CLAIMS_FAILED", retryable=True
            )
