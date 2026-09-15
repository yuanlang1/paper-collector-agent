from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.infrastructure.oss import (
    AliyunOssObjectStore,
    MarkdownObject,
    OssObjectStore,
    normalize_pdf_sha256,
)
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.nodes.finalize import failed
from app.llm.model_factory import create_validated_structured_chat_model
from app.rag.retrieval.paper_content_corpus import PaperContentCorpusReader


EXTRACT_STUDIES_PROMPT = """
你是一名严谨的学术综述研究者。请从一篇已提供全文中逐项抽取研究事实。

只使用全文明确支持的信息；没有明确报告时写 "not_reported"，不要推断。
每个字段的 evidence_quotes 必须是全文中的逐字原文，用于后续程序校验。
研究设计、样本、方法、结果和局限均须分别抽取。
不要生成引用编号、参考文献、论文之外的事实或研究建议。
""".strip()
EVIDENCE_MAP_MAX_CHARS = 12_000
STUDY_FIELD_NAMES = (
    "design",
    "sample",
    "methods",
    "results",
    "limitations",
)


class StudyField(BaseModel):
    summary: str = Field(min_length=1, max_length=1200)
    evidence_quotes: list[str] = Field(default_factory=list, max_length=3)


class ExtractedStudy(BaseModel):
    design: StudyField
    sample: StudyField
    methods: StudyField
    results: StudyField
    limitations: StudyField


class ExtractStudiesNode:
    def __init__(
        self,
        *,
        artifact_store: LocalArtifactStore | None = None,
        model: Any | None = None,
        oss_store: OssObjectStore | None = None,
        corpus_reader: PaperContentCorpusReader | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.model = model or create_validated_structured_chat_model(
            ExtractedStudy,
            temperature=0,
        )
        self.oss_store = oss_store or AliyunOssObjectStore()
        self.corpus_reader = corpus_reader or PaperContentCorpusReader()

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        if state.get("study_records_artifact_ref"):
            return {
                "stage": "generating_framework",
                "status": "running",
            }
        if state.get("stage") != "extracting_studies":
            return failed("extract_studies called in invalid stage")

        try:
            corpus = await self.artifact_store.read_json_uri(
                state["corpus_artifact_ref"]
            )
            papers = corpus["papers"]
            if not isinstance(papers, list):
                raise ValueError("corpus papers is invalid")

            records = []
            failures = []
            warnings = list(state.get("warnings", []))
            for paper in papers:
                if not isinstance(paper, dict):
                    raise ValueError("corpus paper is invalid")
                try:
                    source, fulltext = await self._read_fulltext(paper)
                    extracted = await self._extract(
                        state=state,
                        paper=paper,
                        fulltext=fulltext,
                    )
                    records.append(
                        {
                            "paper_id": str(paper["paper_id"]),
                            "title": str(paper.get("title") or ""),
                            "source": source,
                            **extracted.model_dump(mode="json"),
                        }
                    )
                except Exception as exc:
                    paper_id = str(paper.get("paper_id") or "unknown")
                    failures.append(
                        {
                            "paper_id": paper_id,
                            "error": str(exc) or exc.__class__.__name__,
                        }
                    )
                    warnings.append(
                        f"Study extraction failed for paper_id={paper_id}: {exc}"
                    )

            expected_paper_ids = [
                str(paper_id)
                for paper_id in state["paper_ids_snapshot"]
            ]
            extracted_paper_ids = {
                record["paper_id"]
                for record in records
            }
            failed_paper_ids = {
                failure["paper_id"]
                for failure in failures
            }
            failures.extend(
                {
                    "paper_id": paper_id,
                    "error": "paper is missing from extracted study records",
                }
                for paper_id in expected_paper_ids
                if (
                    paper_id not in extracted_paper_ids
                    and paper_id not in failed_paper_ids
                )
            )
            if failures:
                return await self._failed_extraction(
                    state=state,
                    expected_paper_ids=expected_paper_ids,
                    extracted_paper_ids=extracted_paper_ids,
                    failures=failures,
                    warnings=warnings,
                )

            evidence_map = self._evidence_map(records)
            artifact = await self.artifact_store.write_json(
                run_id=state["run_id"],
                step_key="extract_studies",
                source="task_review",
                kind="task_review_study_records_json",
                count=len(records),
                payload={
                    "task_id": state["task_id"],
                    "paper_ids_snapshot": state["paper_ids_snapshot"],
                    "studies": records,
                    "evidence_map": evidence_map,
                },
            )
        except Exception as exc:
            return failed(f"study extraction failed: {exc}")

        return {
            "study_records_artifact_ref": artifact.artifact_uri,
            "warnings": warnings,
            "stage": "generating_framework",
            "status": "running",
            "error": None,
        }

    async def _failed_extraction(
        self,
        *,
        state: Mapping[str, Any],
        expected_paper_ids: list[str],
        extracted_paper_ids: set[str],
        failures: list[dict[str, str]],
        warnings: list[str],
    ) -> dict[str, Any]:
        report = await self.artifact_store.write_json(
            run_id=state["run_id"],
            step_key="extract_studies",
            source="task_review",
            kind="task_review_study_extraction_report_json",
            count=len(failures),
            payload={
                "task_id": state["task_id"],
                "expected_paper_ids": expected_paper_ids,
                "extracted_paper_ids": sorted(extracted_paper_ids),
                "failures": failures,
            },
        )
        return failed(
            "study extraction did not cover every task paper",
            warnings=warnings,
            study_extraction_report_artifact_ref=report.artifact_uri,
        )

    async def _read_fulltext(
        self,
        paper: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        pdf_sha256 = normalize_pdf_sha256(paper.get("oss_name"))
        if pdf_sha256 is not None:
            try:
                markdown = await self.oss_store.get_markdown(
                    pdf_sha256=pdf_sha256,
                )
            except Exception:
                markdown = None
            if markdown is not None and markdown.content.strip():
                return self._markdown_source(markdown, pdf_sha256), markdown.content

        documents = await self.corpus_reader.read([str(paper["paper_id"])])
        if not documents:
            raise ValueError("Markdown and Qdrant full text are unavailable")

        content = "\n\n".join(
            document.page_content
            for document in documents
            if document.page_content.strip()
        )
        if not content:
            raise ValueError("Qdrant full text is empty")

        return {
            "kind": "qdrant_chunk_fallback",
            "content_sha256": self._sha256(content),
            "chunk_ids": [
                str(document.metadata["chunk_id"])
                for document in documents
            ],
        }, content

    async def _extract(
        self,
        *,
        state: Mapping[str, Any],
        paper: dict[str, Any],
        fulltext: str,
    ) -> ExtractedStudy:
        # ponytail: keep one full-text call so summaries and quotations share one
        # source; introduce section-level reduction only if real papers exceed context.
        result = await self.model.ainvoke(
            [
                SystemMessage(content=EXTRACT_STUDIES_PROMPT),
                HumanMessage(
                    content=json.dumps(
                        {
                            "topic": state["topic"],
                            "output_language": state["language"],
                            "paper": {
                                "paper_id": paper["paper_id"],
                                "title": paper.get("title", ""),
                            },
                            "fulltext": fulltext,
                        },
                        ensure_ascii=False,
                    )
                ),
            ]
        )
        extracted = (
            result
            if isinstance(result, ExtractedStudy)
            else ExtractedStudy.model_validate(result)
        )
        invalid_quotes = [
            quote
            for field_name in STUDY_FIELD_NAMES
            for quote in getattr(
                extracted,
                field_name,
            ).evidence_quotes
            if quote not in fulltext
        ]
        if invalid_quotes:
            raise ValueError("extraction returned evidence outside full text")
        return extracted

    @staticmethod
    def _markdown_source(
        markdown: MarkdownObject,
        pdf_sha256: str,
    ) -> dict[str, str]:
        return {
            "kind": "oss_markdown",
            "pdf_sha256": pdf_sha256,
            "object_name": markdown.object_name,
            "content_sha256": markdown.content_sha256,
        }

    @staticmethod
    def _evidence_map(records: list[dict[str, Any]]) -> list[dict[str, str]]:
        if not records:
            return []

        per_card_budget = (
            EVIDENCE_MAP_MAX_CHARS - len(records) - 1
        ) // len(records)
        cards = [
            ExtractStudiesNode._evidence_card(record, per_card_budget)
            for record in records
        ]
        if len(ExtractStudiesNode._json(cards)) > EVIDENCE_MAP_MAX_CHARS:
            raise ValueError("study evidence map exceeds its prompt budget")
        return cards

    @staticmethod
    def _evidence_card(
        record: dict[str, Any],
        budget: int,
    ) -> dict[str, str]:
        card = {"paper_id": record["paper_id"]}
        if len(ExtractStudiesNode._json(card)) > budget:
            raise ValueError("too many papers for the study evidence map")

        # ponytail: use deterministic clipping to share one bounded profile with
        # both downstream prompts; switch to map-reduce only for richer coverage.
        fixed_text = "\n".join(
            f"{field_name}: "
            for field_name in STUDY_FIELD_NAMES
        )
        empty_card = {**card, "evidence": ""}
        available = budget - len(ExtractStudiesNode._json(empty_card))
        if available <= len(fixed_text):
            return card

        summary_budget = (available - len(fixed_text)) // len(STUDY_FIELD_NAMES)
        summaries = {
            field_name: ExtractStudiesNode._clip(
                str(record[field_name]["summary"]),
                summary_budget,
            )
            for field_name in STUDY_FIELD_NAMES
        }
        evidence = "\n".join(
            f"{field_name}: {summaries[field_name]}"
            for field_name in STUDY_FIELD_NAMES
        )
        card["evidence"] = evidence
        while evidence and len(ExtractStudiesNode._json(card)) > budget:
            evidence = ExtractStudiesNode._clip(
                evidence,
                len(evidence) - 1,
            )
            card["evidence"] = evidence
        return card

    @staticmethod
    def _clip(value: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(value) <= max_chars:
            return value
        return f"{value[:max_chars - 1]}…" if max_chars > 1 else "…"

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _sha256(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
