from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.nodes.finalize import failed
from app.llm.graph.workflows.review_generate.nodes.generate_framework import (
    ReviewFramework,
)
from app.llm.model_factory import create_validated_structured_chat_model


CLAIMS_PROMPT = """
你是一位资深学术研究者，擅长将文献综述框架拆解为结构化的“论点–证据”计划。

【任务】
根据给定的“章节提纲”，为每个章节生成详尽的待验证论点列表。
每条论点将在后续由RAG系统从固定的任务论文集中检索全文证据。

输入中的 evidence_map 概括了固定论文集的逐篇抽取结果。只可据此规划待验证
论点，不得添加 evidence_map 之外的作者、论文、方法、发现、争议或研究空白。

【章节提纲】
由输入中的 framework 提供。

【输出要求】
1. 严格返回输出 Schema 所要求的结构化内容，不要添加额外说明。
2. 必须覆盖 Framework 中的全部章节。
3. 每个章节生成 3–6 条论点。
4. 所有章节合计应尽量覆盖以下分析维度；但不要要求每个章节都覆盖全部维度：
   - 概念边界、理论框架与分析维度；
   - 方法、技术路径、证据类型或评价视角；
   - 研究对象、应用场景或比较维度；
   - 潜在的一致性、差异性、局限性或未解决问题；
   - 不同研究之间可能存在的互补、矛盾或继承关系。
5. 每条 Claim 必须包含：
   - claim_id：全局递增的唯一编号，格式为 claim_001、claim_002……
   - text：使用指定输出语言撰写的论点候选。应具体、可验证且具有学术分析价值。
   - rag_query：3–8 个英文词构成的精确学术短语。
   - claim_type：只能是 descriptive、comparative、critical、gap 之一。
   - evidence_requirement：只能是 fulltext 或 multiple_fulltext。
   - section_id：所属 Framework 章节的 section_id。
   - section_title：所属 Framework 章节的 title。
6. rag_query 仅用于后续在固定 task.paper_ids_snapshot 范围内执行内部 RAG。
   它不是外部检索请求，不得要求搜索新论文、扩展论文集或指定数据库。
7. 不得捏造作者、论文标题、年份、数值、具体研究结果、历史事实、共识、
   争议、局限性或研究空白。
8. 不得生成引用标记、参考文献或 [[REF_x]]。
9. rag_query 应是简洁的英文名词短语，不得使用：
   - 完整句子；
   - 问句或命令句；
   - Boolean 检索语法；
   - 数据库、搜索引擎名称；
   - “find papers”“search literature”等外部检索指令。
10. 优先使用 Framework 中对应章节的 retrieval_hints 构造 rag_query，
    并避免不同章节生成完全重复的 rag_query。

【输出格式示例】
{
  "sections": [
    {
      "section_id": "background",
      "section_title": "研究背景与核心概念",
      "claims": [
        {
          "claim_id": "claim_001",
          "text": "该领域的研究可从核心概念边界、问题定义及其相互关系等维度展开分析。",
          "rag_query": "topic core concepts problem definition",
          "claim_type": "descriptive",
          "evidence_requirement": "fulltext",
          "section_id": "background",
          "section_title": "研究背景与核心概念"
        },
        {
          "claim_id": "claim_002",
          "text": "不同概念界定可能对应不同的研究对象、分析层级和证据需求。",
          "rag_query": "topic conceptual boundaries analytical levels",
          "claim_type": "comparative",
          "evidence_requirement": "multiple_fulltext",
          "section_id": "background",
          "section_title": "研究背景与核心概念"
        }
      ]
    }
  ]
}
""".strip()

REVISE_CLAIMS_PROMPT = """
    你是一位学术综述编辑。请根据反思报告修订指定的 Claim。

    规则：
    1. 仅修改 revision_targets 中的 Claim，不新增章节或 Claim。
    2. 保持 claim_id、section_id 和 section_title 不变。
    3. 将过宽或过强的论断收缩为可由固定任务论文集证据支撑的候选论断。
    4. 可以调整 rag_query 和 evidence_requirement，但不得引入新论文、新事实或引用。
    5. 仅返回结构化输出。
""".strip()


ClaimType = Literal[
    "descriptive",
    "comparative",
    "critical",
    "gap",
]

EvidenceRequirement = Literal[
    "fulltext",
    "multiple_fulltext",
]


class Claim(BaseModel):
    claim_id: str = Field(pattern = r"^claim_\d{3,}$")
    text: str = Field(min_length = 10, max_length = 500)
    rag_query: str = Field(min_length = 3, max_length = 160)
    claim_type: ClaimType
    evidence_requirement: EvidenceRequirement
    section_id: str
    section_title: str


class SectionClaims(BaseModel):
    section_id: str
    section_title: str
    claims: list[Claim] = Field(
        min_length = 3,
        max_length = 6,
    )


class ClaimsPlan(BaseModel):
    sections: list[SectionClaims]


class RevisedClaims(BaseModel):
    claims: list[Claim]

class GenerateClaimsNode:
    def __init__(
        self,
        *,
        artifact_store: LocalArtifactStore | None = None,
        model: Any | None = None,
        revision_model: Any | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.model = model or create_validated_structured_chat_model(
            ClaimsPlan,
            temperature = 0,
        )
        self.revision_model = revision_model or create_validated_structured_chat_model(
            RevisedClaims,
            temperature = 0,
        )

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        revise_claim_ids = state.get("revise_claim_ids", [])

        if state.get("claims_artifact_ref") and not revise_claim_ids:
            return {
                "stage": "retrieving_evidence",
                "status": "running",
            }

        try:
            if revise_claim_ids:
                claims_payload, reflection_report = await asyncio.gather(
                    self.artifact_store.read_json_uri(
                        state["claims_artifact_ref"]
                    ),
                    self.artifact_store.read_json_uri(
                        state["reflection_report_artifact_ref"]
                    ),
                )

                current_claims = ClaimsPlan.model_validate(
                    claims_payload["claims"]
                )
                targets = [
                    claim
                    for section in current_claims.sections
                    for claim in section.claims
                    if claim.claim_id in revise_claim_ids
                ]

                result = await self.revision_model.ainvoke(
                    [
                        SystemMessage(content = REVISE_CLAIMS_PROMPT),
                        HumanMessage(
                            content = json.dumps(
                                {
                                    "topic": state["topic"],
                                    "output_language": state["language"],
                                    "revision_targets": [
                                        claim.model_dump(mode = "json")
                                        for claim in targets
                                    ],
                                    "reflection_report": reflection_report,
                                },
                                ensure_ascii = False,
                            )
                        ),
                    ]
                )

                revised_claims = (
                    result
                    if isinstance(result, RevisedClaims)
                    else RevisedClaims.model_validate(result)
                )
                revised_by_id = {
                    claim.claim_id: claim
                    for claim in revised_claims.claims
                }

                if set(revised_by_id) != {claim.claim_id for claim in targets}:
                    raise ValueError(
                        "claim revision must return exactly the requested claims"
                    )

                claims_plan = ClaimsPlan(
                    sections = [
                        SectionClaims(
                            section_id = section.section_id,
                            section_title = section.section_title,
                            claims = [
                                revised_by_id.get(
                                    claim.claim_id,
                                    claim,
                                )
                                for claim in section.claims
                            ],
                        )
                        for section in current_claims.sections
                    ]
                )
                step_key = f"revise_claims_{state['reflection_round']}"
                affected_section_ids = [
                    claim.section_id
                    for claim in targets
                ]
            else:
                framework_payload, studies_payload = await asyncio.gather(
                    self.artifact_store.read_json_uri(
                        state["framework_artifact_ref"]
                    ),
                    self.artifact_store.read_json_uri(
                        state["study_records_artifact_ref"]
                    ),
                )
                framework = ReviewFramework.model_validate(
                    framework_payload["framework"]
                )
                evidence_map = studies_payload["evidence_map"]
                if not isinstance(evidence_map, list):
                    raise ValueError("study evidence map is invalid")

                result = await self.model.ainvoke(
                    [
                        SystemMessage(content = CLAIMS_PROMPT),
                        HumanMessage(
                            content = json.dumps(
                                {
                                    "topic": state["topic"],
                                    "output_language": state["language"],
                                    "review_type": state["review_type"],
                                    "framework": framework.model_dump(
                                        mode = "json"
                                    ),
                                    "evidence_map": evidence_map,
                                },
                                ensure_ascii = False,
                            )
                        ),
                    ]
                )

                claims_plan = (
                    result
                    if isinstance(result, ClaimsPlan)
                    else ClaimsPlan.model_validate(result)
                )
                step_key = "generate_claims"
                affected_section_ids = []

            artifact = await self.artifact_store.write_json(
                run_id = state["run_id"],
                step_key = step_key,
                source = "task_review",
                kind = "task_review_claims_json",
                count = sum(
                    len(section.claims)
                    for section in claims_plan.sections
                ),
                payload = {
                    "task_id": state["task_id"],
                    "framework_hash": state["framework_hash"],
                    "claims": claims_plan.model_dump(mode = "json"),
                },
            )

        except Exception as exc:
            return failed(f"claim generation failed: {exc}")

        retrieve_claim_ids = list(
            dict.fromkeys(
                [
                    *state.get("retrieve_claim_ids", []),
                    *revise_claim_ids,
                ]
            )
        )

        return {
            "claims_artifact_ref": artifact.artifact_uri,
            "revise_claim_ids": [],
            "retrieve_claim_ids": retrieve_claim_ids,
            "render_section_ids": list(
                dict.fromkeys(
                    [
                        *state.get("render_section_ids", []),
                        *affected_section_ids,
                    ]
                )
            ),
            "stage": "retrieving_evidence",
            "status": "running",
            "error": None,
        }
