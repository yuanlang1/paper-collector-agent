from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.nodes.generate_framework import (
    ReviewFramework,
)
from app.llm.model_factory import create_validated_structured_chat_model


MAX_CHUNKS_PER_CLAIM = 3


RENDER_SECTION_PROMPT = """
你是一位专业的学术综述写作者。

根据给定的章节信息、待写论点和正文证据片段，撰写一节连贯的文献综述。

严格规则：
1. 只能使用输入中提供的 Claim 和正文 chunk。
2. 任何具体事实、比较、研究发现或结论，都必须使用对应的 [[REF_id]]
   引用锚点。
3. 不得编造作者、年份、论文、方法、数值、结论或引用。
4. 不得使用未提供的 [[REF_id]]。
5. 不要逐篇论文罗列；应围绕 Claim 综合不同论文的证据，说明一致、
   差异、互补或冲突。
6. 对证据存在冲突的内容，应呈现为研究分歧，不能写成单一结论。
7. 没有证据的 Claim 已被排除，不得自行补充。
8. 不要输出章节标题；直接输出正文。
9. 输出内容不应为了凑篇幅而重复、扩写或添加无证据内容。

返回结构化输出，不要添加额外解释。
""".strip()


class SectionDraft(BaseModel):
    text: str = Field(min_length = 1)
    citation_map: dict[str, int] = Field(default_factory = dict)
    used_claim_ids: list[str] = Field(default_factory = list)
    omitted_claim_ids: list[str] = Field(default_factory = list)
    summary: str = Field(min_length = 1, max_length = 500)


class RenderSectionsNode:
    def __init__(
        self,
        *,
        artifact_store: LocalArtifactStore | None = None,
        model: Any | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        
        self.model = model or create_validated_structured_chat_model(
            SectionDraft,
            temperature = 0,
        )

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            (
                framework_payload,
                claims_payload,
                evidence_payload,
            ) = await asyncio.gather(
                self.artifact_store.read_json_uri(
                    state["framework_artifact_ref"]
                ),
                self.artifact_store.read_json_uri(
                    state["claims_artifact_ref"]
                ),
                self.artifact_store.read_json_uri(
                    state["evidence_ledger_artifact_ref"]
                ),
            )

            framework = ReviewFramework.model_validate(framework_payload["framework"])

            claims_by_section = {
                section["section_id"]: section["claims"]
                for section in claims_payload["claims"]["sections"]
            }
            evidence_by_claim = {
                item["claim_id"]: item
                for item in evidence_payload["claims"]
            }

            section_refs = dict(state.get("section_draft_artifact_refs", {}))
            target_section_ids = (
                state.get("render_section_ids")
                or [
                    section.section_id
                    for section in framework.sections
                ]
            )

            previous_summary = ""

            for section in framework.sections:
                section_id = section.section_id

                if section_id not in target_section_ids:
                    if section_id in section_refs:
                        previous_draft = await self.artifact_store.read_json_uri(
                            section_refs[section_id]
                        )
                        previous_summary = previous_draft.get(
                            "summary",
                            previous_summary,
                        )
                    continue

                claims = claims_by_section.get(section_id, [])
                supported_claims, omitted_claim_ids = (
                    self._build_supported_claims(
                        claims,
                        evidence_by_claim,
                    )
                )

                if not supported_claims:
                    draft_data = {
                        "text": "",
                        "citation_map": {},
                        "used_claim_ids": [],
                        "omitted_claim_ids": [
                            claim["claim_id"]
                            for claim in claims
                        ],
                        "summary": "该章节缺少可用于写作的正文证据。",
                    }
                else:
                    result = await self.model.ainvoke(
                        [
                            SystemMessage(
                                content = RENDER_SECTION_PROMPT,
                            ),
                            HumanMessage(
                                content = json.dumps(
                                    {
                                        "section": {
                                            "section_id": section_id,
                                            "title": section.title,
                                            "description": (
                                                section.description
                                            ),
                                        },
                                        "previous_section_summary": (
                                            previous_summary
                                        ),
                                        "claims_with_evidence": (
                                            supported_claims
                                        ),
                                        "omitted_claim_ids": (
                                            omitted_claim_ids
                                        ),
                                    },
                                    ensure_ascii = False,
                                )
                            ),
                        ]
                    )

                    draft = (
                        result
                        if isinstance(result, SectionDraft)
                        else SectionDraft.model_validate(result)
                    )

                    draft_data = draft.model_dump(mode = "json")
                    draft_data["omitted_claim_ids"] = list(
                        dict.fromkeys(
                            [
                                *omitted_claim_ids,
                                *draft_data["omitted_claim_ids"],
                            ]
                        )
                    )

                artifact = await self.artifact_store.write_json(
                    run_id = state["run_id"],
                    step_key = f"render_section_{section_id}",
                    source = "task_review",
                    kind = "task_review_section_draft_json",
                    count = len(draft_data["used_claim_ids"]),
                    payload = {
                        "task_id": state["task_id"],
                        "framework_hash": state["framework_hash"],
                        "section_id": section_id,
                        "title": section.title,
                        **draft_data,
                    },
                )

                section_refs[section_id] = artifact.artifact_uri
                previous_summary = draft_data["summary"]

        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"section rendering failed: {exc}",
            }

        return {
            "section_draft_artifact_refs": section_refs,
            "render_section_ids": [],
            "stage": "assembling_review",
            "status": "running",
            "error": None,
        }

    def _build_supported_claims(
        self,
        claims: list[dict[str, Any]],
        evidence_by_claim: dict[str, dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        supported_claims = []
        omitted_claim_ids = []

        for claim in claims:
            evidence = evidence_by_claim.get(
                claim["claim_id"],
                {},
            )
            snippets = evidence.get("chunk_snippets", [])

            paper_ids = {
                str(snippet["paper_id"])
                for snippet in snippets
            }

            needs_multiple_papers = (
                claim["evidence_requirement"] == "multiple_fulltext"
            )

            if not snippets or ( needs_multiple_papers and len(paper_ids) < 2):
                omitted_claim_ids.append(claim["claim_id"])
                continue

            supported_claims.append(
                {
                    "claim_id": claim["claim_id"],
                    "text": claim["text"],
                    "claim_type": claim["claim_type"],
                    "evidence_requirement": claim["evidence_requirement"],
                    "evidence": [
                        {
                            "anchor": f"[[REF_{snippet['paper_id']}]]",
                            "paper_id": snippet["paper_id"],
                            "title": snippet.get("title", ""),
                            "chunk_id": snippet["chunk_id"],
                            "chunk_index": snippet.get(
                                "chunk_index"
                            ),
                            "section_path": snippet.get(
                                "section_path",
                                "",
                            ),
                            "page_start": snippet.get("page_start"),
                            "page_end": snippet.get("page_end"),
                            "page_numbers": snippet.get("page_numbers", []),
                            "text": snippet["text"],
                        }
                        for snippet in snippets[
                            :MAX_CHUNKS_PER_CLAIM
                        ]
                    ],
                }
            )

        return supported_claims, omitted_claim_ids

