from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.nodes.generate_framework import ReviewFramework
from app.llm.model_factory import create_structured_chat_model


REF_PATTERN = re.compile(r"\[\[REF_(\d+)\]\]")

ASSEMBLE_REVIEW_PROMPT = """
你是一位学术综述编辑。

请根据已经写好的综述正文，为其生成摘要和结论。

严格规则：
1. 只能概括输入正文已经表达的内容，不得引入新事实、新方法、新论文、新数据或新研究结论。
2. 不得虚构作者、年份、论文标题、引用或 [[REF_id]] 锚点。
3. 摘要应概括综述主题、覆盖范围、主要组织维度和总体讨论内容。
4. 结论应归纳正文已经呈现的主要认识、分歧、局限或待解决问题；只有正文已有依据时才能表达。
5. 如果结论中的具体判断需要引用，请保留正文中已有的 [[REF_id]] 锚点。
6. 使用指定输出语言。
7. 仅返回结构化输出，不添加解释。
""".strip()


class ReviewSynthesis(BaseModel):
    abstract: str
    conclusion: str


class AssembleReviewNode:
    def __init__(
        self,
        *,
        artifact_store: LocalArtifactStore | None = None,
        model: Any | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.model = model or create_structured_chat_model(
            ReviewSynthesis,
            temperature = 0,
        )

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            framework_payload = await self.artifact_store.read_json_uri(
                state["framework_artifact_ref"]
            )
            corpus_payload = await self.artifact_store.read_json_uri(
                state["corpus_artifact_ref"]
            )
            framework = ReviewFramework.model_validate(framework_payload["framework"])
            section_refs = state["section_draft_artifact_refs"]

            draft_payloads = await asyncio.gather(
                *[
                    self.artifact_store.read_json_uri(
                        section_refs[section.section_id]
                    )
                    for section in framework.sections
                    if section.section_id in section_refs
                ]
            )
        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"review assembly failed: {exc}",
            }

        drafts_by_section = {
            draft["section_id"]: draft
            for draft in draft_payloads
        }

        sections = []
        omitted_sections = []

        for section in framework.sections:
            draft = drafts_by_section.get(section.section_id)
            text = (draft or {}).get("text", "").strip()

            if not text:
                omitted_sections.append(
                    {
                        "section_id": section.section_id,
                        "title": section.title,
                        "reason": "no_rendered_text",
                    }
                )
                continue

            sections.append(
                {
                    "section_id": section.section_id,
                    "title": section.title,
                    "description": section.description,
                    "text": text,
                    "summary": draft["summary"],
                    "used_claim_ids": draft.get("used_claim_ids", []),
                    "omitted_claim_ids": draft.get(
                        "omitted_claim_ids",
                        [],
                    ),
                }
            )

        body_markdown = "\n\n".join(
            f"## {section['title']}\n\n{section['text']}"
            for section in sections
        )

        if sections:
            result = await self.model.ainvoke(
                [
                    SystemMessage(content = ASSEMBLE_REVIEW_PROMPT),
                    HumanMessage(
                        content = json.dumps(
                            {
                                "output_language": state["language"],
                                "review_title": framework.title,
                                "review_scope": framework.scope,
                                "section_summaries": [
                                    {
                                        "section_id": section["section_id"],
                                        "title": section["title"],
                                        "summary": section["summary"],
                                    }
                                    for section in sections
                                ],
                                "review_body": body_markdown,
                            },
                            ensure_ascii = False,
                        )
                    ),
                ]
            )

            synthesis = (
                result
                if isinstance(result, ReviewSynthesis)
                else ReviewSynthesis.model_validate(result)
            )
        else:
            synthesis = ReviewSynthesis(
                abstract = "",
                conclusion = "",
            )

        review_text = "\n\n".join(
            part
            for part in [
                body_markdown,
                synthesis.conclusion.strip(),
            ]
            if part
        )

        citation_paper_ids = list(
            dict.fromkeys(
                REF_PATTERN.findall(review_text)
            )
        )

        papers_by_id = {
            str(paper["paper_id"]): paper
            for paper in corpus_payload["papers"]
        }

        citation_metadata = {
            paper_id: {
                "title": papers_by_id[paper_id]["title"],
                "authors": papers_by_id[paper_id].get(
                    "authors",
                    [],
                ),
                "published_date": papers_by_id[paper_id].get(
                    "published_date"
                ),
                "doi": papers_by_id[paper_id].get("doi"),
            }
            for paper_id in citation_paper_ids
            if paper_id in papers_by_id
        }

        unknown_citation_paper_ids = [
            paper_id
            for paper_id in citation_paper_ids
            if paper_id not in papers_by_id
        ]

        review_draft = {
            "task_id": state["task_id"],
            "framework_hash": state["framework_hash"],
            "title": framework.title,
            "scope": framework.scope,
            "abstract": synthesis.abstract.strip(),
            "sections": sections,
            "body_markdown": body_markdown,
            "conclusion": synthesis.conclusion.strip(),
            "citation_paper_ids": citation_paper_ids,
            "citation_metadata": citation_metadata,
            "unknown_citation_paper_ids": unknown_citation_paper_ids,
            "omitted_sections": omitted_sections,
        }

        try:
            artifact = await self.artifact_store.write_json(
                run_id = state["run_id"],
                step_key = "assemble_review",
                source = "task_review",
                kind = "task_review_draft_json",
                count = len(sections),
                payload = review_draft,
            )
        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"review draft persistence failed: {exc}",
            }

        return {
            "review_draft_artifact_ref": artifact.artifact_uri,
            "stage": "reflecting_review",
            "status": "running",
            "error": None,
        }

