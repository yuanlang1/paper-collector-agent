from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.contracts import (
    content_hash,
    evidence_requirement_met,
    invoke,
    save,
    verified_pack,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import failed
from app.llm.provider import ChatClient, ModelOptions


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

    每个 quote 必须：
    * 逐字来自 original_evidence；
    * 使用正确的 paper_id 和 chunk_id；
    * 标记为 `supports` 或 `opposes`；
    * 不得改写、拼接或虚构原文。

    优先选择能够直接证明或反驳 claim 的最小充分证据，不要为了数量罗列弱相关片段。

    ## Revision 规则

    Claim 必须服从 Evidence。
    当 `status != supported` 时，必须针对当前 `claim_id` 产生一个 `target_type="claim"` 的修订项。
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

    不要把同一 claim 的限定条件、补充说明或轻微改写拆成新 claim。

    `add_claim.target_id` 必须是当前 section_id。

    当当前 claim 为 `supported` 时，可以没有 revision，也可以仅包含 `add_claim`。

    ## 边界
    你只能：
    * 核验当前 claim；
    * 修改当前 claim；
    * 基于现有证据建议同章节新增一个独立 claim。

    不得修改 framework、section、abstract 或 conclusion；不得重新设计研究问题；不得要求重新 retrieval。
    最终判断应遵循：
    **Evidence 决定 Claim 可以说到什么程度，而不是为了保留 Claim 去解释或选择 Evidence。**

    以下仅为结构示例；quote 必须逐字取自实际 original_evidence，
    ID 必须来自实际输入，勿照抄内容：
    supported：
    ```json
    {
      "status": "supported",
      "evidence": [{
        "paper_id": "paper_001",
        "chunk_id": "chunk_001",
        "quote": "原文支持片段",
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
        "target_id": "claim_001",
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
        "paper_id": "paper_001",
        "chunk_id": "chunk_001",
        "quote": "原文支持片段",
        "relation": "supports"
      }],
      "reason": "当前论点已被支持，且存在未覆盖的独立发现。",
      "revisions": [{
        "target_type": "add_claim",
        "target_id": "section_001",
        "issue": "存在未覆盖的独立发现。",
        "required_change": "新增一个仅陈述该发现的候选论点。",
        "acceptance_criteria": "新增论点由现有证据直接支持且不重复已有论点。"
      }]
    }
    ```
"""
)


class EvidenceQuote(BaseModel):
    paper_id: str = Field(description="证据所属的当前任务论文 ID。")
    chunk_id: str = Field(description="证据在该论文中的原文片段 ID。")
    quote: str = Field(
        min_length=1,
        description="逐字摘录自输入 original_evidence 的最小充分原文，不得改写或拼接。",
    )
    relation: Literal["supports", "opposes"] = Field(
        description="该原文片段与当前 Claim 的关系：supports 直接支持，opposes 直接反驳。",
    )


class ClaimRevision(BaseModel):
    target_type: Literal["claim", "add_claim"] = Field(
        description="修改类型：claim 修订当前论点，add_claim 在当前章节新增独立候选论点。",
    )
    target_id: str = Field(
        description="claim 类型必须为当前 claim_id；add_claim 类型必须为当前 section_id。",
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
        description="真正参与支持或反驳判断的关键逐字原文证据。",
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


class VerifyClaimsNode:
    def __init__(self, *, artifact_store=None, model=None, chat: ChatClient | None = None):
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.chat = chat or ChatClient()
        self.model = model or self.chat.structured(Verdict, options=ModelOptions(temperature=0))

    async def _revision_snapshot(self, state):
        values = []
        for key in ("claims_artifact_ref", "evidence_ledger_artifact_ref"):
            value = await self.artifact_store.read_json_uri(state[key]) if state.get(key) else None
            if value and key == "evidence_ledger_artifact_ref":
                value = [
                    {"claim_id": item["claim_id"], "chunk_snippets": item["chunk_snippets"]}
                    for item in value["claims"]
                ]
            if value and key == "claims_artifact_ref":
                value = [
                    {"claim_id": item["claim_id"], "claim_hash": item["claim_hash"]}
                    for item in value["claims"]
                ]
            values.append(value)
        return content_hash(values)

    async def _schedule_claim_revision(self, state, items, *, claim_ids, section_ids):
        allowed = {"claim": set(claim_ids), "add_claim": set(section_ids)}
        if any(
            item["target_type"] not in allowed
            or item["target_id"] not in allowed[item["target_type"]]
            for item in items
        ):
            raise ValueError("revision contains an unknown target")
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
        if not items or current == state.get("revision_before"):
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
            framework = (
                await self.artifact_store.read_json_uri(state["framework_artifact_ref"])
            )["framework"]
            sections = {section["section_id"]: section for section in framework["sections"]}
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
                    and not cached.get("revisions")
                ):
                    verdict = cached
                else:
                    verdict = await invoke(
                        self.model,
                        Verdict,
                        CLAIM_VERIFICATION_PROMPT,
                        {
                            "claim": claim,
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
                            "original_evidence": evidence,
                        },
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
                if verdict["status"] == "supported" and not evidence_requirement_met(
                    claim, verdict
                ):
                    verdict.update(
                        status="insufficient",
                        reason=(
                            f'{verdict["reason"]} '
                            "Supporting sources do not meet the claim requirement."
                        ),
                        revisions=[
                            *[
                                item
                                for item in verdict["revisions"]
                                if item["target_type"] == "add_claim"
                            ],
                            dict(
                                target_type="claim",
                                target_id=claim["claim_id"],
                                issue="Supporting source count is below the claim requirement",
                                required_change=(
                                    "Narrow, revise, or withdraw the claim to match "
                                    "the available evidence"
                                ),
                                acceptance_criteria=(
                                    "The revised claim's scope and source requirement "
                                    "match the evidence"
                                ),
                            )
                        ],
                    )
                if verdict["status"] == "supported":
                    verified_pack(claim, verdict)
                elif not any(item["target_type"] == "claim" for item in verdict["revisions"]):
                    raise ValueError("unsupported verdict needs a current claim revision")
                for item in verdict["revisions"]:
                    if item["target_type"] == "claim" and item["target_id"] != claim["claim_id"]:
                        raise ValueError("claim revision must target the current claim")
                    if (
                        item["target_type"] == "add_claim"
                        and item["target_id"] != claim["section_id"]
                    ):
                        raise ValueError("added claim must target the current section")
                verdicts.append(verdict)
                revisions.extend(verdict["revisions"])
            supported = {item["claim_id"] for item in verdicts if item["status"] == "supported"}
            ref = await save(
                self.artifact_store,
                state,
                "claim_verification",
                {"claims": verdicts},
            )
            update = {"claim_verification_artifact_ref": ref}
            if revisions:
                if (
                    state["reflection_round"]
                    >= state["max_reflection_rounds"] - 1
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
                result = await self._schedule_claim_revision(
                    {**state, **update},
                    revisions,
                    claim_ids=[claim["claim_id"] for claim in claims],
                    section_ids=list(sections),
                )
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
        except Exception as exc:
            return failed(
                f"claim verification failed: {exc}",
                error_code="VERIFICATION_FAILED",
                retryable=True,
            )
