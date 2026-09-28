from pydantic import BaseModel, Field

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.contracts import (
    CandidateClaim,
    claim_hash,
    invoke,
    revisions_for,
    save,
    validate_claims,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import failed
from app.llm.provider import ChatClient, ModelOptions


CLAIM_GENERATION_PROMPT = """
    你是一名学术综述论点设计者。围绕当前 section 的 discussion_questions 比较和综合 studies，
    生成具体、可核验的候选论点，不按论文逐篇罗列。每个论点应服务于至少一个讨论问题，
    且不得与 existing_candidates 重复。论文画像不是原文证据，不得预设一致性、优越性或因果结论，
    也不可新增论文。

    保留分歧、条件和局限。comparative 论点必须使用 multiple_fulltext；
    每条 retrieval_queries 独立表达该论点的检索意图，合起来覆盖支持、反证和适用条件。
    revisions 非空时，只执行其中针对当前章节的新增要求。使用 output_language。

    以下仅为格式示例；使用实际论文 ID 和章节内容，勿照抄：
    ```json
    {
    "claims": [
        {
        "text": "在给定研究条件下，两种方法的结果差异可能与测量设置有关。",
        "candidate_paper_ids": ["paper_001", "paper_002"],
        "retrieval_queries": [
            "方法 A 与方法 B 的结果比较",
            "测量设置对结果的影响",
            "方法比较中的相反结果"
        ],
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
    条件、对象或结论不可兼容时拆分为多个独立论点。不能成立的非核心候选填写 withdrawn_reason。
    不可改变 section，不得虚构；ID、section_id 和哈希由程序处理。

    每条检索查询独立表达修改后的论点，并可面向支持、反证或条件改写。
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
    拆分：
    ```json
    {
    "claims": [
        {
        "text": "条件甲下结果呈现关联。",
        "candidate_paper_ids": ["paper_001"],
        "retrieval_queries": ["条件甲 结果关联"],
        "claim_type": "descriptive",
        "evidence_requirement": "fulltext",
        "withdrawn_reason": "",
        "section_id": ""
        },
        {
        "text": "条件乙下结果存在差异。",
        "candidate_paper_ids": ["paper_002"],
        "retrieval_queries": ["条件乙 结果差异"],
        "claim_type": "descriptive",
        "evidence_requirement": "fulltext",
        "withdrawn_reason": "",
        "section_id": ""
        }
    ]
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
        description="替换原论点的一条或多条候选论点；撤回也用一条带 withdrawn_reason 的论点表达。",
    )


class GenerateClaimsNode:
    def __init__(
        self,
        *,
        artifact_store=None,
        model=None,
        revision_model=None,
        chat: ChatClient | None = None,
    ):
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
        result = await invoke(
            self.model,
            ClaimsPlan,
            CLAIM_GENERATION_PROMPT,
            {
                "section": section,
                "studies": relevant,
                "existing_candidates": [claim["text"] for claim in existing],
                "revisions": additions,
                "output_language": state["language"],
            },
        )
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
                evidence = await self.artifact_store.read_json_uri(
                    state["evidence_ledger_artifact_ref"]
                )
                result = await invoke(
                    self.revision_model,
                    ClaimRevisionPlan,
                    CLAIM_REVISION_PROMPT,
                    {
                        "old_claim": claim,
                        "section": sections[claim["section_id"]],
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
                    else [
                        f'{claim["claim_id"]}_{index}'
                        for index in range(1, len(replacements) + 1)
                    ]
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
