import asyncio
import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.database import SessionLocal
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.contracts import (
    CandidateClaim,
    UNTRUSTED_INPUT_GUARD,
    claim_hash,
    review_batch,
    revisions_for,
    save,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import failed
from app.llm.provider import ChatClient, ModelOptions
from app.services import llm_setting_service


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
        user_id: int | None = None,
    ):
        self.artifact_store = artifact_store or LocalArtifactStore()
        self._model_lock = asyncio.Lock()
        self.user_id = user_id
        self.model = model
        self.revision_model = revision_model

    async def _load_models(self):
        async with SessionLocal() as db:
            config = await llm_setting_service.get_small_llm_config(db, user_id=self.user_id)
        client = ChatClient(runtime_config=config)
        options = ModelOptions(temperature=0)
        if self.model is None:
            self.model = client.structured(ClaimsPlan, options=options)
        if self.revision_model is None:
            self.revision_model = client.structured(ClaimRevisionPlan, options=options)

    async def _ensure_model(self, *, revise=False):
        if (self.revision_model if revise else self.model) is None:
            async with self._model_lock:
                if (self.revision_model if revise else self.model) is None:
                    await self._load_models()

    async def _generate_for_section(
        self, state, section, studies, existing, additions, claim_budget=MAX_CLAIMS_PER_SECTION
    ):
        relevant = [
            {
                "paper_id": str(paper["paper_id"]),
                "title": paper["title"],
                "paper_profile": paper["paper_profile"],
            }
            for paper in studies
            if str(paper["paper_id"]) in section["relevant_paper_ids"]
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

    async def prepare(self, state):
        framework = (await self.artifact_store.read_json_uri(state["framework_artifact_ref"]))["framework"]
        sections = {section["section_id"]: section for section in framework["sections"]}
        studies = (await self.artifact_store.read_json_uri(state["corpus_artifact_ref"]))["papers"]
        payload = (
            await self.artifact_store.read_json_uri(state["claims_artifact_ref"])
            if state.get("claims_artifact_ref")
            else {"claims": [], "unanswered": []}
        )
        claims = [claim for claim in payload["claims"] if claim["section_id"] in sections]
        unanswered = [item for item in payload["unanswered"] if item["section_id"] in sections]
        targets = {item["target_id"] for item in revisions_for(state, "claim")}
        if targets:
            evidence_by_claim = {
                item["claim_id"]: item
                for item in (await self.artifact_store.read_json_uri(
                    state["evidence_ledger_artifact_ref"]
                ))["claims"]
            }
            jobs = [
                {
                    "item_id": claim["claim_id"],
                    "revision": {
                        "old_claim": claim,
                        "section": sections[claim["section_id"]],
                        "output_language": state["language"],
                        "revisions": revisions_for(state, "claim", claim["claim_id"]),
                        "evidence": evidence_by_claim[claim["claim_id"]],
                    },
                }
                for claim in claims
                if claim["claim_id"] in targets
            ]
            if jobs:
                await self._ensure_model(revise=True)
                return review_batch(
                    jobs, mode="revise", sections=framework["sections"], studies=studies,
                    claims=claims, unanswered=unanswered, first_generation=not payload["claims"],
                )
        return await self._plan_generation(
            state, framework["sections"], studies, claims, unanswered, not payload["claims"],
        )

    async def _plan_generation(self, state, sections, studies, claims, unanswered, first_generation):
        budgets = claim_budgets(sections, claims)
        jobs = []
        sections_to_generate = []
        for section in sections:
            section_id = section["section_id"]
            additions = revisions_for(state, "add_claim", section_id)
            if not first_generation and not additions:
                continue
            sections_to_generate.append(section_id)
            if budgets[section_id]:
                jobs.append({
                    "item_id": section_id,
                    "language": state["language"],
                    "section": section,
                    "studies": [
                        paper for paper in studies
                        if str(paper["paper_id"]) in section["relevant_paper_ids"]
                    ],
                    "existing": [claim for claim in claims if claim["section_id"] == section_id],
                    "additions": additions,
                    "budget": budgets[section_id],
                })
        if jobs:
            await self._ensure_model()
        return review_batch(
            jobs, mode="generate", sections=sections, sections_to_generate=sections_to_generate,
            budgets=budgets, claims=claims, unanswered=unanswered,
        )

    async def run_item(self, item):
        if "revision" in item:
            await self._ensure_model(revise=True)
            result = await self.revision_model.ainvoke([
                SystemMessage(content=CLAIM_REVISION_PROMPT + UNTRUSTED_INPUT_GUARD),
                HumanMessage(content=json.dumps(item["revision"], ensure_ascii=False)),
            ])
            return result.model_dump(mode="json")["claims"][0]
        await self._ensure_model()
        candidates, reason = await self._generate_for_section(
            item, item["section"], item["studies"], item["existing"], item["additions"], item["budget"],
        )
        return {"candidates": candidates, "reason": reason}

    async def collect(self, state, results, failure):
        if failure:
            return self.failure(state, failure)
        context = state["batch_context"]
        if context["mode"] == "revise":
            claims = []
            for claim in context["claims"]:
                if claim["claim_id"] in results:
                    claim = {
                        **results[claim["claim_id"]],
                        "claim_id": claim["claim_id"],
                        "section_id": claim["section_id"],
                    }
                    claim["claim_hash"] = claim_hash(claim)
                claims.append(claim)
            batch = await self._plan_generation(
                state, context["sections"], context["studies"], claims,
                context["unanswered"], context["first_generation"],
            )
            if batch["batch_jobs"]:
                return batch
            context = batch["batch_context"]
            results = {}
        return await self._save_claims(state, context, results)

    async def _save_claims(self, state, context, results):
        claims = list(context["claims"])
        unanswered = context["unanswered"]
        budgets = context["budgets"]
        sections_to_generate = set(context["sections_to_generate"])
        next_index = max(
            (int(claim["claim_id"].removeprefix("claim_")) for claim in claims),
            default=0,
        ) + 1
        warnings = list(state["warnings"])
        for section in context["sections"]:
            section_id = section["section_id"]
            if section_id not in sections_to_generate:
                continue
            budget = budgets[section_id]
            if budget:
                candidates = results[section_id]["candidates"]
                reason = results[section_id]["reason"]
            else:
                candidates, reason = [], "Claim budget exhausted for this section."
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
                claim = {**claim, "claim_id": claim_id, "section_id": section_id}
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

    @staticmethod
    def failure(state, error):
        return failed(
            f"claim generation failed: {error['error']}",
            error_code="CLAIMS_FAILED", retryable=True,
        )
