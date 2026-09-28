from pydantic import BaseModel, Field

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.citations import citation_anchor_ids
from app.llm.graph.workflows.review_generate.contracts import (
    content_hash,
    invoke,
    revisions_for,
    save,
    verified_pack,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import failed
from app.llm.provider import ChatClient, ModelOptions


SECTION_DRAFT_PROMPT = """
你是一名严谨的学术综述写作者。基于 claims_with_evidence 为当前章节写作，使用 output_language。

写作目标：围绕 section 的 discussion_questions 比较和综合研究，而不是逐篇复述。只表达当前已核验 Claim 与证据能够支持的内容；保留研究之间的分歧、条件、范围和局限。

规则：
1. claims_with_evidence 非空时，arguments 必须覆盖其中每个 claim_id 至少一次；只有直接相关的论点才能放入同一个 Argument。
2. 每个 Argument 的 claim_ids 和 evidence_ids 必须来自输入。具体事实、比较和结论使用对应的
   [[REF_论文ID]] 锚点；锚点只能引用 evidence_ids 所属论文。
3. 不得引入输入之外的论文、事实、方法、数据、因果关系或结论。不得将条件性、相关性或单项发现扩大为普遍结论。
4. revisions 非空时，完成其中的 required_change；不要沿用旧草稿中无法由当前证据支持的表述。
5. summary 只概括本节 arguments 已表达的已核验内容，不得新增判断或引用。claims_with_evidence 非空时，insufficient_notice 必须为空。
6. claims_with_evidence 为空时，arguments 必须为空；仅在 insufficient_notice 中说明材料不足以回答本节问题，且不得使用引用或给出肯定性结论。

有证据时的输出示例：
```json
{
  "arguments": [
    {
      "text": "在给定条件下，研究结果呈现差异 [[REF_1]] [[REF_2]]。",
      "claim_ids": ["claim_001"],
      "evidence_ids": ["evidence_1", "evidence_2"]
    }
  ],
  "summary": "结果差异与研究条件相关。",
  "insufficient_notice": ""
}
```

材料不足时的输出示例：
```json
{
  "arguments": [],
  "summary": "当前材料不足以形成比较。",
  "insufficient_notice": "现有已核验证据不足以回答本节问题。"
}
```

仅返回 SectionDraft 的结构化输出，不添加解释。
""".strip()


class Argument(BaseModel):
    """章节正文中绑定 Claim 与证据引用的一个论证单元。"""

    text: str = Field(
        min_length=1,
        description="基于已核验证据写成的正文论证，应包含对应 [[REF_论文ID]] 锚点。",
    )
    claim_ids: list[str] = Field(
        min_length=1,
        description="该论证直接覆盖的输入已核验 Claim ID。",
    )
    evidence_ids: list[str] = Field(
        min_length=1,
        description="直接支撑该论证并限定可用引用论文的输入 evidence_id。",
    )


class SectionDraft(BaseModel):
    """单个综述章节的结构化草稿。"""

    arguments: list[Argument] = Field(
        default_factory=list,
        description="按论证单元组织的章节正文；无已核验证据时必须为空。",
    )
    summary: str = Field(
        min_length=1,
        max_length=500,
        description="本节 arguments 的简短衔接性概述，不得引入新的判断或事实。",
    )
    insufficient_notice: str = Field(
        default="",
        description="无已核验证据时说明无法回答章节问题的原因；有证据时必须为空。",
    )


class RenderSectionsNode:
    def __init__(self, *, artifact_store=None, model=None, chat: ChatClient | None = None):
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.model = model or (chat or ChatClient()).structured(
            SectionDraft,
            options=ModelOptions(temperature=0),
        )

    async def __call__(self, state):
        refs = dict(state.get("section_draft_artifact_refs", {}))
        try:
            framework = (await self.artifact_store.read_json_uri(state["framework_artifact_ref"]))[
                "framework"
            ]
            claims = {
                claim["claim_id"]: claim
                for claim in (
                    await self.artifact_store.read_json_uri(state["claims_artifact_ref"])
                )["claims"]
            }
            verdicts = {
                item["claim_id"]: item
                for item in (
                    await self.artifact_store.read_json_uri(
                        state["claim_verification_artifact_ref"]
                    )
                )["claims"]
            }
            refs = {
                key: ref
                for key, ref in refs.items()
                if key in {section["section_id"] for section in framework["sections"]}
            }
            revision_targets = {
                item["target_id"]
                for item in state.get("revision_items", [])
                if item["target_type"] == "section"
            }
            previous_summary = ""
            for section in framework["sections"]:
                section_claims = [
                    claim
                    for claim in claims.values()
                    if claim["section_id"] == section["section_id"]
                    and claim["claim_id"] in verdicts
                    and verdicts[claim["claim_id"]]["status"] == "supported"
                ]
                packages = [
                    {"claim": claim, "evidence": verified_pack(claim, verdicts[claim["claim_id"]])}
                    for claim in section_claims
                ]
                revisions = revisions_for(state, "section", section["section_id"])
                signature = content_hash(
                    [
                        framework["title"],
                        framework["scope"],
                        section,
                        packages,
                        state["language"],
                        previous_summary,
                    ]
                )
                old_ref = state.get("section_draft_artifact_refs", {}).get(section["section_id"])
                old = await self.artifact_store.read_json_uri(old_ref) if old_ref else None
                if old and section["section_id"] not in revision_targets and (
                    revision_targets or old.get("input_hash") == signature
                ):
                    refs[section["section_id"]] = old_ref
                    previous_summary = old["summary"]
                    continue

                if packages:
                    result = await invoke(
                        self.model,
                        SectionDraft,
                        SECTION_DRAFT_PROMPT,
                        {
                            "output_language": state["language"],
                            "review_title": framework["title"],
                            "review_scope": framework["scope"],
                            "section": section,
                            "claims_with_evidence": packages,
                            "old_draft": old if revisions else None,
                            "revisions": revisions,
                            "previous_section_summary": previous_summary,
                        },
                    )
                    if result["insufficient_notice"]:
                        raise ValueError(
                            "evidence-backed section cannot contain an insufficiency notice"
                        )
                    claim_ids = {item["claim"]["claim_id"] for item in packages}
                    evidence = {
                        item["evidence_id"]: item for pack in packages for item in pack["evidence"]
                    }
                    for argument in result["arguments"]:
                        if (
                            not set(argument["claim_ids"]) <= claim_ids
                            or not set(argument["evidence_ids"]) <= evidence.keys()
                        ):
                            raise ValueError(
                                "section argument references unknown claim or evidence"
                            )
                        papers = {evidence[eid]["paper_id"] for eid in argument["evidence_ids"]}
                        anchors = set(citation_anchor_ids(argument["text"]))
                        if not anchors or not anchors <= papers:
                            raise ValueError("section citations are not bound to supplied evidence")
                        for claim_id in argument["claim_ids"]:
                            claim_evidence = {
                                item["evidence_id"]: item
                                for item in verdicts[claim_id]["evidence"]
                            }
                            support = {
                                claim_evidence[evidence_id]["paper_id"]
                                for evidence_id in argument["evidence_ids"]
                                if evidence_id in claim_evidence
                                and claim_evidence[evidence_id]["relation"] == "supports"
                            }
                            required = (
                                2
                                if claims[claim_id]["evidence_requirement"] == "multiple_fulltext"
                                else 1
                            )
                            if len(support & anchors) < required:
                                raise ValueError("argument omitted required claim sources")
                    arguments = result["arguments"]
                    used = {
                        claim_id
                        for argument in arguments
                        for claim_id in argument["claim_ids"]
                    }
                    if used != claim_ids:
                        raise ValueError("section draft omitted verified claims")
                    text = "\n\n".join(item["text"] for item in arguments)
                    summary = result["summary"]
                else:
                    result = await invoke(
                        self.model,
                        SectionDraft,
                        SECTION_DRAFT_PROMPT,
                        {
                            "output_language": state["language"],
                            "review_title": framework["title"],
                            "review_scope": framework["scope"],
                            "section": section,
                            "claims_with_evidence": [],
                            "old_draft": old if revisions else None,
                            "revisions": revisions,
                            "previous_section_summary": previous_summary,
                        },
                    )
                    arguments = result["arguments"]
                    if (
                        arguments
                        or not result["insufficient_notice"]
                        or citation_anchor_ids(result["insufficient_notice"])
                    ):
                        raise ValueError(
                            "insufficient-evidence section must contain only an uncited notice"
                        )
                    text = result["insufficient_notice"]
                    summary = result["summary"]

                draft = {
                    "section_id": section["section_id"],
                    "title": section["title"],
                    "framework_hash": state["framework_hash"],
                    "input_hash": signature,
                    "claim_hashes": {
                        claim["claim_id"]: claim["claim_hash"]
                        for claim in section_claims
                    },
                    "arguments": arguments,
                    "text": text,
                    "summary": summary,
                    "used_claim_ids": [claim["claim_id"] for claim in section_claims],
                    "omitted_claim_ids": [],
                }
                refs[section["section_id"]] = await save(
                    self.artifact_store, state, "section_draft", draft
                )
                previous_summary = draft["summary"]
            return {
                "section_draft_artifact_refs": refs,
                "revision_items": [
                    item
                    for item in state.get("revision_items", [])
                    if item["target_type"] != "section"
                ],
                "stage": "assembling_review",
                "status": "running",
            }
        except Exception as exc:
            return failed(
                f"section rendering failed: {exc}",
                section_draft_artifact_refs=refs,
                error_code="RENDER_FAILED",
                retryable=True,
            )
