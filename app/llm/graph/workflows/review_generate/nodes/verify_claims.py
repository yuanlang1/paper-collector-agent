import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphBubbleUp
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.config import settings
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.contracts import (
    UNTRUSTED_INPUT_GUARD,
    content_hash,
    evidence_requirement_met,
    required_source_count,
    review_batch,
    save,
    supporting_paper_ids,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import failed
from app.llm.provider import ChatClient, ModelOptions
from app.services import llm_setting_service
from app.llm.provider.structured_output import StructuredOutputError


CLAIM_VERIFICATION_PROMPT = ("""
    你是一名严格的学术论点核验员。
    你的任务是判断当前 claim 是否被输入的 original_evidence 真实支持。只能依据提供的原文证据，不得使用外部知识，不得因为主题相关、片段数量较多或表述看似合理就判定支持。

    重点核验：
    1. **对象与范围**：研究对象、方法、数据集、人群、时间范围等是否一致；
    2. **限定条件**：claim 是否遗漏原文中的重要条件或适用范围；
    3. **数值与比较**：数值、指标、基线和比较条件是否一致，不能把“更高”扩大为“显著优于”；
    4. **因果强度**：不能把相关、关联、观察结果扩大为因果关系；
    5. **概括强度**：不能把单篇或少量研究扩大为“普遍”“主流”“大多数”“领域趋势”；
    6. **反证与分歧**：必须同时考虑 supports 和 opposes 证据，不能只选择支持当前 claim 的片段；
    7. **可比性**：实验条件、指标或研究对象不可比时，不得直接得出比较结论。

    ## status 判断
    * `supported`：现有证据能够直接支持当前 claim 的核心表述、范围和强度。
    * `mixed`：部分证据支持、部分证据反对，或不同条件下结论不同；当前 claim 需要收缩、增加限定条件或改为呈现分歧。
    * `contradicted`：现有关键证据与当前 claim 的核心结论直接冲突，当前 claim 不应保持原表述。
    * `insufficient`：现有证据不足以证明或反驳 claim，例如缺少直接依据、证据范围过窄或无法支撑其概括强度。

    证据不足不等于 claim 为假；不得因没有找到相关片段而断言领域不存在某种研究。

    ## Evidence 输出
    `evidence` 只保留真正参与判断的关键原文。
    输入是经过相关性筛选的证据片段，不代表论文全部内容或完整领域证据；未发现支持或反例时，
    只能判断当前证据不足，不得据此断言相关研究不存在。

    每个 evidence 必须：
    * 使用 original_evidence 中的 source_index；
    * 标记为 `supports` 或 `opposes`；
    * 简短说明该原文如何支持或反驳 claim；可概括，不得改变原文含义。

    优先选择能够直接证明或反驳 claim 的最小充分证据，不要为了数量罗列弱相关片段。

    ## 独立来源要求
    original_evidence 的 paper_index 只用于识别独立论文，不能输出到 evidence。
    `fulltext` 至少需要一篇 supports 的论文；`multiple_fulltext` 至少需要两篇 supports 的不同论文。
    同一篇论文的多个片段只算一篇，片段内转述的其他论文也不增加独立来源数，opposes 不计入支持篇数。
    支持论文不足时必须返回 `insufficient`，并给出当前 claim 的修订建议。

    ## Revision 规则

    Claim 必须服从 Evidence。
    当 `status != supported` 时，必须产生一个 `target_type="claim"` 的修订项；程序会绑定当前 claim。
    修订应根据证据选择：

    * 收缩论点范围；
    * 补充必要限定条件；
    * 降低结论强度；
    * 将单向结论改为分歧性表述；
    * 拆分过宽论点；
    * 证据无法承载时撤回该论点。

    不得要求重新检索或补充证据。

    `required_change` 描述需要如何修改，不要直接生成一段正文。

    `acceptance_criteria` 必须明确说明修改后的 claim 满足什么条件才能被现有证据支持。

    ## add_claim

    只有当现有 evidence 清楚显示一个：

    * 能回答当前章节至少一个讨论问题；
    * 能被现有证据独立支持；
    * 且没有被 section_claims 已有论点覆盖

    的重要独立论点时，才允许产生 `target_type="add_claim"`。

    限定条件、反例、补充说明及已有 claim 的轻微改写不构成新增论点。

    `add_claim` 会由程序绑定当前 section。

    当当前 claim 为 `supported` 时，可以没有 revision，也可以仅包含 `add_claim`。

    ## 边界
    你只能：
    * 核验当前 claim；
    * 修改当前 claim；
    * 基于现有证据建议同章节新增一个独立 claim。

    不得修改 framework、section、abstract 或 conclusion；不得重新设计研究问题；不得要求重新 retrieval。
    最终判断应遵循：
    **Evidence 决定 Claim 可以说到什么程度，而不是为了保留 Claim 去解释或选择 Evidence。**

    source_index 必须来自实际输入。若输入包含 validation_feedback，修正所述问题后返回完整 JSON。
    以下仅为结构示例，勿照抄内容：
    supported：
    ```json
    {
      "status": "supported",
      "evidence": [{
        "source_index": 1,
        "assessment": "原文在给定条件下支持该论点。",
        "relation": "supports"
      }],
      "reason": "证据直接支持该论点的范围。",
      "revisions": []
    }
    ```
    insufficient：
    ```json
    {
      "status": "insufficient",
      "evidence": [],
      "reason": "现有片段不能支持原论点的概括强度。",
      "revisions": [{
        "target_type": "claim",
        "issue": "范围超出证据。",
        "required_change": "收缩为已见条件下的描述。",
        "acceptance_criteria": "修改后仅陈述现有片段直接支持的条件性结果。"
      }]
    }
    ```
    supported 且新增同章节论点：
    ```json
    {
      "status": "supported",
      "evidence": [{
        "source_index": 1,
        "assessment": "原文在给定条件下支持该论点。",
        "relation": "supports"
      }],
      "reason": "当前论点已被支持，且存在未覆盖的独立发现。",
      "revisions": [{
        "target_type": "add_claim",
        "issue": "存在未覆盖的独立发现。",
        "required_change": "新增一个仅陈述该发现的候选论点。",
        "acceptance_criteria": "新增论点由现有证据直接支持且不重复已有论点。"
      }]
    }
    ```
"""
)


class EvidenceQuote(BaseModel):
    source_index: int = Field(
        ge=1,
        description="本次 original_evidence 中从 1 开始的证据编号。",
    )
    assessment: str = Field(
        min_length=1,
        description="概括该原文如何支持或反驳 Claim，不得改变原文含义。",
    )
    relation: Literal["supports", "opposes"] = Field(
        description="该原文片段与当前 Claim 的关系：supports 直接支持，opposes 直接反驳。",
    )


class ClaimRevision(BaseModel):
    target_type: Literal["claim", "add_claim"] = Field(
        description="修改类型：claim 修订当前论点，add_claim 在当前章节新增独立候选论点。",
    )
    issue: str = Field(
        min_length=1,
        description="由现有原文证据确认的具体问题或遗漏。",
    )
    required_change: str = Field(
        min_length=1,
        description="后续 Claim 生成节点必须执行的明确修改动作，不直接写综述正文。",
    )
    acceptance_criteria: str = Field(
        min_length=1,
        description="修改后在现有证据下应满足的可核验条件。",
    )


class Verdict(BaseModel):
    status: Literal["supported", "mixed", "contradicted", "insufficient"] = Field(
        description="核验状态：supported 可写作；mixed、contradicted、insufficient 均需修订当前 Claim。",
    )
    evidence: list[EvidenceQuote] = Field(
        description="真正参与支持或反驳判断的关键原文证据及其概括。",
    )
    reason: str = Field(
        min_length=1,
        description="说明证据如何支持、反驳或不足以支撑当前 Claim 的判断理由。",
    )
    revisions: list[ClaimRevision] = Field(
        default_factory=list,
        description="必须执行的 Claim 修订或同章节新增请求；supported 只能为空或仅含 add_claim。",
    )

    @model_validator(mode="after")
    def validate_revisions(self):
        if self.status == "supported":
            if any(item.target_type != "add_claim" for item in self.revisions):
                raise ValueError("supported verdict can only add claims")
        elif not any(item.target_type == "claim" for item in self.revisions):
            raise ValueError("unsupported verdict must revise the current claim")
        return self


class VerificationProtocolError(ValueError):
    pass


class VerifyClaimsNode:
    def __init__(
        self,
        *,
        artifact_store=None,
        model=None,
        user_id: int | None = None,
    ):
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.user_id = user_id
        self.model = model

    @staticmethod
    def _sources(evidence, titles):
        return {
            index: {
                "paper_id": item["paper_id"],
                "chunk_id": item["chunk_id"],
                "paper_title": titles.get(item["paper_id"], ""),
                "section_path": item.get("section_path"),
                "text": item["text"],
                **{
                    key: item.get(key)
                    for key in ("page_start", "page_end", "page_numbers")
                },
            }
            for index, item in enumerate(evidence, start=1)
        }

    @staticmethod
    def _bind_verdict(result, claim, sources):
        verdict = dict(result)
        evidence = []
        for item in verdict["evidence"]:
            source = sources.get(item["source_index"])
            if source is None:
                raise VerificationProtocolError(
                    f"source_index={item['source_index']} is not available; "
                    f"choose one of {list(sources)}"
                )
            evidence.append(
                {
                    **source,
                    "relation": item["relation"],
                    "assessment": item["assessment"],
                    "char_start": 0,
                    "char_end": len(source["text"]),
                    "evidence_id": content_hash(
                        [source["paper_id"], source["chunk_id"], source["text"]]
                    )[:20],
                }
            )
        verdict["evidence"] = evidence
        verdict["revisions"] = [
            {
                **item,
                "target_id": (
                    claim["claim_id"]
                    if item["target_type"] == "claim"
                    else claim["section_id"]
                ),
            }
            for item in verdict["revisions"]
        ]
        if verdict["status"] == "supported" and not evidence_requirement_met(
            claim, verdict
        ):
            support_count = len(supporting_paper_ids(verdict))
            required_count = required_source_count(claim)
            verdict["status"] = "insufficient"
            verdict["reason"] = (
                f"{verdict['reason']} Program check found {support_count} supporting "
                f"papers; {required_count} required."
            )
            verdict["revisions"].append(
                {
                    "target_type": "claim",
                    "target_id": claim["claim_id"],
                    "issue": "Supporting paper count is below the claim requirement.",
                    "required_change": (
                        "Narrow the claim to the available evidence. For a single-study "
                        "result, use a descriptive claim with a fulltext requirement; "
                        "otherwise withdraw it."
                    ),
                    "acceptance_criteria": (
                        "The revised text, claim type, and evidence requirement match "
                        "the available supporting papers."
                    ),
                }
            )
        return verdict

    async def _verify_claim(self, *, claim, section, section_claims, sources):
        paper_indices = {}
        for source in sources.values():
            if source["paper_id"] not in paper_indices:
                paper_indices[source["paper_id"]] = len(paper_indices) + 1
        payload = {
            "claim": claim,
            "section": section,
            "section_claims": section_claims,
            "original_evidence": [
                {
                    "source_index": index,
                    "paper_index": paper_indices[source["paper_id"]],
                    "paper_title": source["paper_title"],
                    "section_path": source["section_path"],
                    "text": source["text"],
                }
                for index, source in sources.items()
            ],
        }
        for attempt in range(2):
            try:
                result = (await self.model.ainvoke([
                    SystemMessage(content=CLAIM_VERIFICATION_PROMPT + UNTRUSTED_INPUT_GUARD),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                ])).model_dump(mode="json")
                return self._bind_verdict(result, claim, sources)
            except (StructuredOutputError, ValidationError, VerificationProtocolError) as exc:
                if attempt:
                    raise VerificationProtocolError(str(exc)) from exc
                payload["validation_feedback"] = str(exc)

    async def _revision_snapshot(self, state):
        claims = await self.artifact_store.read_json_uri(state["claims_artifact_ref"])
        ledger = await self.artifact_store.read_json_uri(state["evidence_ledger_artifact_ref"])
        return content_hash([
            [{"claim_id": item["claim_id"], "claim_hash": item["claim_hash"]} for item in claims["claims"]],
            [{"claim_id": item["claim_id"], "chunk_snippets": item["chunk_snippets"]} for item in ledger["claims"]],
        ])

    async def _schedule_claim_revision(self, state, items):
        current = await self._revision_snapshot(state)
        ref = await save(
            self.artifact_store,
            state,
            "revision_plan",
            {
                "items": items,
                "before": current,
                "reflection_round": state["reflection_round"] + 1,
            },
        )
        update = {"revision_plan_artifact_ref": ref, "revision_items": items}
        if current == state.get("revision_before"):
            return failed(
                "revision produced no actionable progress",
                **update,
                error_code="REVIEW_NO_PROGRESS",
            )
        return {
            **update,
            "revision_before": current,
            "reflection_round": state["reflection_round"] + 1,
            "stage": "generating_claims",
            "status": "running",
        }

    async def _ensure_model(self):
        if self.model is None:
            config = await llm_setting_service.get_small_llm_config(user_id=self.user_id)
            if self.model is None:
                self.model = ChatClient(runtime_config=config).structured(
                    Verdict, max_attempts=1, options=ModelOptions(temperature=0),
                )

    async def prepare(self, state):
        claims = (await self.artifact_store.read_json_uri(state["claims_artifact_ref"]))["claims"]
        ledger = {
            item["claim_id"]: item
            for item in (await self.artifact_store.read_json_uri(state["evidence_ledger_artifact_ref"]))["claims"]
        }
        framework = (await self.artifact_store.read_json_uri(state["framework_artifact_ref"]))["framework"]
        corpus = await self.artifact_store.read_json_uri(state["corpus_artifact_ref"])
        sections = {section["section_id"]: section for section in framework["sections"]}
        titles = {
            str(paper["paper_id"]): paper.get("title", "")
            for paper in corpus["papers"]
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
        verdicts, jobs, claim_ids = {}, [], []
        for claim in claims:
            if claim["withdrawn_reason"]:
                continue
            claim_ids.append(claim["claim_id"])
            if ledger[claim["claim_id"]]["claim_hash"] != claim["claim_hash"]:
                raise ValueError("evidence ledger belongs to a previous claim version")
            evidence = ledger[claim["claim_id"]]["chunk_snippets"]
            digest = content_hash(evidence)
            cached = previous.get(claim["claim_id"])
            if (
                cached
                and cached["claim_hash"] == claim["claim_hash"]
                and cached["evidence_digest"] == digest
                and not cached.get("revisions")
            ):
                verdicts[claim["claim_id"]] = cached
            else:
                jobs.append(
                    {
                        "item_id": claim["claim_id"],
                        "claim": claim,
                        "digest": digest,
                        "section": sections[claim["section_id"]],
                        "section_claims": [
                            {
                                key: item[key]
                                for key in (
                                    "claim_id",
                                    "text",
                                    "claim_type",
                                    "withdrawn_reason",
                                )
                            }
                            for item in claims
                            if item["section_id"] == claim["section_id"]
                        ],
                        "sources": self._sources(evidence, titles),
                    }
                )

        if jobs:
            await self._ensure_model()
        return review_batch(jobs, claim_ids=claim_ids, cached=verdicts)

    async def run_item(self, item):
        await self._ensure_model()
        claim = item["claim"]
        verdict = await self._verify_claim(
            claim=claim, section=item["section"], section_claims=item["section_claims"],
            sources=item["sources"],
        )
        verdict.update(
            claim_id=claim["claim_id"],
            claim_hash=claim["claim_hash"],
            evidence_digest=item["digest"],
        )
        return verdict

    async def collect(self, state, results, failure):
        context = state["batch_context"]
        verdicts = {**context["cached"], **results}
        ordered_verdicts = [
            verdicts[claim_id] for claim_id in context["claim_ids"] if claim_id in verdicts
        ]
        verification_ref = state.get("claim_verification_artifact_ref")
        if ordered_verdicts or not failure:
            verification_ref = await save(
                self.artifact_store, state, "claim_verification", {"claims": ordered_verdicts},
            )
        update = {"claim_verification_artifact_ref": verification_ref}
        if failure:
            return self.failure({**state, **update}, failure)

        revisions = [revision for verdict in ordered_verdicts for revision in verdict["revisions"]]
        supported = any(verdict["status"] == "supported" for verdict in ordered_verdicts)
        if revisions:
            if (
                state["reflection_round"]
                >= settings.TASK_REVIEW_MAX_REFLECTION_ROUNDS - 1
            ):
                return {
                    **update,
                    "revision_items": [],
                    "warnings": state.get("warnings", []) + [
                        "Claim evidence revision limit reached; unsupported claims "
                        "will be represented as material insufficiency."
                    ],
                    "stage": "rendering_sections",
                    "status": "running",
                }
            try:
                result = await self._schedule_claim_revision({**state, **update}, revisions)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                return self.failure({**state, **update}, {"error": str(exc)})
            return {**update, **result}
        return {
            **update,
            "warnings": (
                state.get("warnings", [])
                + [
                    "No claim has verified evidence; "
                    "sections will state material insufficiency."
                ]
                if not supported
                else state.get("warnings", [])
            ),
            "stage": "rendering_sections",
            "status": "running",
        }

    @staticmethod
    def failure(state, error):
        message = error["error"]
        if "item_id" in error:
            message = f"claim_id={error['item_id']}: {message}"
        protocol_error = error.get("protocol_error", False)
        label = "claim verification protocol failed" if protocol_error else "claim verification failed"
        return failed(
            f"{label}: {message}",
            claim_verification_artifact_ref=state.get("claim_verification_artifact_ref"),
            error_code="VERIFICATION_PROTOCOL_FAILED" if protocol_error else "VERIFICATION_FAILED",
            retryable=not protocol_error,
        )
