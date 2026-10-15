import asyncio
import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.contracts import (
    content_hash,
    revisions_for,
    save,
    verified_pack,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import failed
from app.llm.provider import ChatClient, ModelOptions


SECTION_DRAFT_PROMPT = """
你是一名严谨的学术综述写作者。基于 claims_with_evidence 为当前章节写作，使用 output_language。

写作目标：围绕 section 的 discussion_questions 比较和综合研究，而不是逐篇复述。只表达当前已核验 Claim 与证据能够支持的内容；保留研究之间的分歧、条件、范围和局限。
section_outline 是固定的综述目录；本节只讨论当前 section 的范围，避免重复相邻章节，
不使用“如上文所述”等依赖具体前文内容的指代。

规则：
1. claims_with_evidence 非空时，arguments 必须覆盖其中每个 claim_id 至少一次；只有直接相关的论点才能放入同一个 Argument。
2. 每个 evidence 都有本章节内有效的 source_index。evidence_indices 只能选择输入明确给出的 source_index，
   不得按列表位置猜测编号，也不得使用旧草稿或其他章节的编号。
3. 每个 Argument 的 claim_ids 和 evidence_indices 必须来自输入。对每个覆盖的 Claim，所选 supports
   证据必须满足 evidence_requirement；multiple_fulltext 需要两篇不同论文的 supports 证据，opposes 不计入支持数量。
4. text 只写正文，不得写 Pandoc 引用、REF 锚点、数字标签或作者年份引用；程序会根据 evidence_indices 插入引用。
5. 不得引入输入之外的论文、事实、方法、数据、因果关系或结论。不得将条件性、相关性或单项发现扩大为普遍结论。
6. revisions 非空时，完成其中的 required_change；不要沿用旧草稿中无法由当前证据支持的表述。
7. summary 只概括本节 arguments 已表达的已核验内容，不得新增判断或引用。claims_with_evidence 非空时，insufficient_notice 必须为空。
8. claims_with_evidence 为空时，arguments 必须为空；仅在 insufficient_notice 中说明材料不足以回答章节问题，且不得使用引用或给出肯定性结论。

有证据时的输出示例：
```json
{
  "arguments": [
    {
      "text": "在给定条件下，研究结果呈现差异。",
      "claim_ids": ["claim_001"],
      "evidence_indices": [1, 2]
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
    text: str = Field(
        min_length=1,
        description="基于已核验证据写成的正文论证，不得自行写入引用。",
    )
    claim_ids: list[str] = Field(
        min_length=1,
        description="该论证直接覆盖的输入已核验 Claim ID。",
    )
    evidence_indices: list[int] = Field(
        min_length=1,
        description="直接支撑该论证的输入 evidence.source_index。",
    )


class SectionDraft(BaseModel):
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
    def __init__(
        self,
        *,
        artifact_store=None,
        model=None,
        chat: ChatClient | None = None,
        max_concurrency: int = 4,
    ):
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.max_concurrency = max_concurrency
        self.model = model or (chat or ChatClient()).structured(
            SectionDraft,
            options=ModelOptions(temperature=0),
        )

    @staticmethod
    def _citation_text(text, paper_ids):
        citations = "; ".join(f"@paper-{paper_id}" for paper_id in paper_ids)
        return f"{text.rstrip()} [{citations}]"

    @staticmethod
    def _model_packages(packages, sources):
        source_indexes = {
            item["evidence_id"]: index for index, item in sources.items()
        }
        return [
            {
                "claim": {
                    key: pack["claim"][key]
                    for key in ("claim_id", "text", "evidence_requirement")
                },
                "evidence": [
                    {
                        "source_index": source_indexes[item["evidence_id"]],
                        **{
                            key: item[key]
                            for key in (
                                "paper_title",
                                "section_path",
                                "text",
                                "relation",
                                "assessment",
                            )
                        },
                    }
                    for item in pack["evidence"]
                ],
            }
            for pack in packages
        ]

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
            section_outline = [
                {
                    key: section[key]
                    for key in ("section_id", "title", "description")
                }
                for section in framework["sections"]
            ]
            semaphore = asyncio.Semaphore(self.max_concurrency)

            tasks = [
                asyncio.create_task(self._render_section(
                    state, section, framework, claims, verdicts,
                    section_outline, revision_targets, semaphore,
                ))
                for section in framework["sections"]
            ]
            try:
                for task in asyncio.as_completed(tasks):
                    section_id, ref = await task
                    refs[section_id] = ref
            except (asyncio.CancelledError, Exception):
                for task in tasks:
                    task.cancel()
                for result in await asyncio.gather(*tasks, return_exceptions=True):
                    if not isinstance(result, BaseException):
                        section_id, ref = result
                        refs[section_id] = ref
                raise
            refs = {
                section["section_id"]: refs[section["section_id"]]
                for section in framework["sections"]
                if section["section_id"] in refs
            }
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

    async def _render_section(
        self, state, section, framework, claims, verdicts,
        section_outline, revision_targets, semaphore,
    ):
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
                section_outline,
            ]
        )
        old_ref = state.get("section_draft_artifact_refs", {}).get(section["section_id"])
        old = await self.artifact_store.read_json_uri(old_ref) if old_ref else None
        if old and section["section_id"] not in revision_targets and (
            revision_targets or old.get("input_hash") == signature
        ):
            return section["section_id"], old_ref

        evidence = {
            item["evidence_id"]: item
            for pack in packages
            for item in pack["evidence"]
        }
        sources = {index: item for index, item in enumerate(evidence.values(), start=1)}
        payload = {
            "output_language": state["language"],
            "review_title": framework["title"],
            "review_scope": framework["scope"],
            "section_outline": section_outline,
            "section": section,
            "claims_with_evidence": self._model_packages(packages, sources),
            "old_draft": old if revisions else None,
            "revisions": revisions,
        }
        async with semaphore:
            result = (await self.model.ainvoke([
                SystemMessage(content=SECTION_DRAFT_PROMPT + UNTRUSTED_INPUT_GUARD),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ])).model_dump(mode="json")
        if packages:
            arguments = self._render_arguments(result, packages, sources, claims, verdicts)
            text = "\n\n".join(item["text"] for item in arguments)
        else:
            arguments = result["arguments"]
            if (
                arguments
                or not result["insufficient_notice"]
                or "@paper-" in result["insufficient_notice"]
                or "[[REF_" in result["insufficient_notice"]
            ):
                raise ValueError("insufficient-evidence section must contain only an uncited notice")
            text = result["insufficient_notice"]

        draft = {
            "section_id": section["section_id"],
            "title": section["title"],
            "input_hash": signature,
            "claim_hashes": {
                claim["claim_id"]: claim["claim_hash"]
                for claim in section_claims
            },
            "arguments": arguments,
            "text": text,
            "summary": result["summary"],
            "used_claim_ids": [claim["claim_id"] for claim in section_claims],
            "omitted_claim_ids": [],
        }
        ref = await save(
            self.artifact_store, state, "section_draft", draft
        )
        return section["section_id"], ref

    def _render_arguments(self, result, packages, sources, claims, verdicts):
        if result["insufficient_notice"]:
            raise ValueError(
                "evidence-backed section cannot contain an insufficiency notice"
            )
        claim_ids = {item["claim"]["claim_id"] for item in packages}
        arguments = []
        for argument in result["arguments"]:
            if (
                not set(argument["claim_ids"]) <= claim_ids
                or not set(argument["evidence_indices"]) <= sources.keys()
            ):
                raise ValueError(
                    "section argument references unknown claim or evidence"
                )
            if "@paper-" in argument["text"] or "[[REF_" in argument["text"]:
                raise ValueError("section argument must not contain citations")
            evidence_ids = [
                sources[index]["evidence_id"]
                for index in argument["evidence_indices"]
            ]
            for claim_id in argument["claim_ids"]:
                claim_evidence = {
                    item["evidence_id"]: item
                    for item in verdicts[claim_id]["evidence"]
                }
                support = {
                    claim_evidence[evidence_id]["paper_id"]
                    for evidence_id in evidence_ids
                    if evidence_id in claim_evidence
                    and claim_evidence[evidence_id]["relation"] == "supports"
                }
                required = (
                    2
                    if claims[claim_id]["evidence_requirement"] == "multiple_fulltext"
                    else 1
                )
                if len(support) < required:
                    raise ValueError("argument omitted required claim sources")
            paper_ids = list(
                dict.fromkeys(
                    sources[index]["paper_id"]
                    for index in argument["evidence_indices"]
                )
            )
            arguments.append(
                {
                    "text": self._citation_text(argument["text"], paper_ids),
                    "claim_ids": argument["claim_ids"],
                    "evidence_ids": evidence_ids,
                }
            )
        used = {
            claim_id
            for argument in arguments
            for claim_id in argument["claim_ids"]
        }
        if used != claim_ids:
            raise ValueError("section draft omitted verified claims")
        return arguments
