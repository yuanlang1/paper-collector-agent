from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.provider import ChatClient, ModelOptions
from pypdf import PdfReader


PDF_FRONT_PAGE_ENRICHMENT_SYSTEM_PROMPT = """
你是学术论文摘要整理员，依据 source_abstract、abstract_source 和 pdf_front_pages 输出原文摘要与整理后摘要。

规则：
- source_abstract 非空且 abstract_source 不是 search_snippet 时，paper_abstract 原样保留来源摘要。
- source_abstract 为空或仅为 search_snippet（检索片段）时，优先从 PDF 提取完整 Abstract，保留原文语言。
- 若来源摘要为空且 PDF 没有独立 Abstract 段落，必须依据可用正文用论文原文语言生成 paper_abstract，并将 paper_abstract_generated 设为 true；不得虚构事实或冒充作者原文。
- 若只有检索片段且无法提取完整摘要，保留片段，不得把片段扩写成虚构的完整原文。
- ai_abstract 是整理后摘要，使用中文概括研究问题、核心方法、主要结果和贡献；原文摘要缺失时，在同一次调用中同时生成 paper_abstract 和 ai_abstract。
- keywords 优先提取 Keywords、Index Terms 或“关键词”段落；未找到时可依据页面文本归纳关键词。
- PDF 文本为空时，仅依据来源摘要整理。没有证据的实验设置、数值和结论不要补写，不能仅凭标题生成摘要。

输出规范：
- 只输出合法 JSON 对象，不要输出 Markdown、代码块或额外说明。
- keywords 为字符串数组，ai_abstract 必须是非空中文文本。

输出样例：
{
  "paper_abstract": "This paper studies retrieval augmented generation.",
  "paper_abstract_generated": false,
  "keywords": ["retrieval augmented generation", "large language models"],
  "ai_abstract": "论文研究检索增强生成方法，并在给定实验设置下报告其效果。"
}
""".strip()


PDF_READ_PAGE_COUNT = 4
MAX_ABSTRACT_CHARS = 8_000
MAX_KEYWORDS = 12
MAX_KEYWORD_CHARS = 1_000
ABSTRACT_ENRICHMENT_CONCURRENCY = 3


class PdfFrontPageEnrichmentResult(BaseModel):
    paper_abstract: str | None = Field(
        default=None,
        max_length=MAX_ABSTRACT_CHARS,
    )
    paper_abstract_generated: bool = Field(
        default=False, description="原文摘要是否由正文归纳生成；直接提取或保留来源摘要时为 false。"
    )
    keywords: list[str] = Field(default_factory=list, max_length=MAX_KEYWORDS)
    ai_abstract: str = Field(min_length=1, max_length=3_000)


def _text(value: Any) -> str | None:
    if value is None:
        return None

    normalized = str(value).strip()
    return normalized or None


def _extract_initial_pdf_text(
    pdf_path: Path,
) -> tuple[str, int]:
    reader = PdfReader(str(pdf_path))
    pages = reader.pages[:PDF_READ_PAGE_COUNT]
    text = "\n".join(
        page.extract_text() or ""
        for page in pages
    )

    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text).strip()

    if not text:
        raise ValueError("PDF 前四页没有可提取的文本层。")

    return text, len(pages)


def _normalize_keywords(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    keywords: list[str] = []
    seen: set[str] = set()

    for item in values:
        for candidate in re.split(r"[,;，；\n|•·]+", str(item or "")):
            normalized = re.sub(r"\s+", " ", candidate).strip(" .:-—")
            key = normalized.casefold()
            if (
                not normalized
                or len(normalized) > 160
                or key in seen
            ):
                continue
            seen.add(key)
            keywords.append(normalized)
            if len(keywords) >= MAX_KEYWORDS:
                return keywords

    return keywords


def _keywords_text(keywords: list[str]) -> str | None:
    value = ", ".join(keywords).strip()
    return value[:MAX_KEYWORD_CHARS] or None


class AbstractEnrichNode:
    def __init__(
        self,
        artifact_store: LocalArtifactStore | None = None,
        model: Any | None = None,
        chat: ChatClient | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        client = chat or ChatClient()
        self.model = model or client.structured(
            PdfFrontPageEnrichmentResult,
            options=ModelOptions(temperature=0),
        )

    async def _enrich_from_pdf_front_pages(
        self,
        initial_pdf_text: str,
        source_abstract: str | None,
        abstract_source: str | None,
    ) -> PdfFrontPageEnrichmentResult:
        result = await self.model.ainvoke(
            [
                SystemMessage(content=PDF_FRONT_PAGE_ENRICHMENT_SYSTEM_PROMPT),
                HumanMessage(
                    content=json.dumps(
                        {
                            "pdf_front_pages": initial_pdf_text,
                            "source_abstract": source_abstract,
                            "abstract_source": abstract_source,
                        },
                        ensure_ascii=False,
                    )
                ),
            ]
        )
        return (
            result
            if isinstance(result, PdfFrontPageEnrichmentResult)
            else PdfFrontPageEnrichmentResult.model_validate(result)
        )

    async def _enrich_paper(
        self,
        paper: dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> tuple[dict[str, int], str | None]:
        counts = {
            "source_abstract": 0,
            "extracted_abstract": 0,
            "missing_abstract": 0,
            "front_pages_read": 0,
            "ai_abstract": 0,
            "source_keyword": 0,
            "extracted_keyword": 0,
            "missing_keyword": 0,
        }
        paper_info = paper["paper_info"]
        pdf_download = paper.get("pdf_download") or {}

        source_abstract = _text(paper_info.get("paper_abstract"))
        source_keywords = _normalize_keywords(paper_info.get("keywords"))
        local_pdf_path = _text(pdf_download.get("local_pdf_path"))
        page_count = 0
        content_error: str | None = None
        extracted_abstract: str | None = None
        extracted_keywords: list[str] = []
        generated = False
        pdf_error = None
        initial_pdf_text = ""

        try:
            async with semaphore:
                try:
                    if not local_pdf_path:
                        raise ValueError("缺少本地 PDF 路径。")
                    initial_pdf_text, page_count = await asyncio.to_thread(
                        _extract_initial_pdf_text, Path(local_pdf_path),
                    )
                    counts["front_pages_read"] += 1
                except Exception as exc:
                    pdf_error = str(exc)
                    if not source_abstract:
                        raise
                enrichment = await self._enrich_from_pdf_front_pages(
                    initial_pdf_text, source_abstract, paper_info.get("abstract_source"),
                )

            extracted_abstract = _text(enrichment.paper_abstract)
            generated = enrichment.paper_abstract_generated
            extracted_keywords = _normalize_keywords(enrichment.keywords)
            ai_abstract = _text(enrichment.ai_abstract)
            if ai_abstract is None:
                raise ValueError("AI 摘要为空。")
            if not source_abstract and not extracted_abstract:
                raise ValueError("缺少来源摘要时，必须同时生成原文摘要。")
            paper_info["ai_abstract"] = ai_abstract
            counts["ai_abstract"] += 1
        except Exception as exc:
            content_error = str(exc)
            paper_info["ai_abstract"] = None

        if source_abstract and (
            paper_info.get("abstract_source") != "search_snippet"
            or not initial_pdf_text
            or not extracted_abstract
            or extracted_abstract == source_abstract
            or generated
        ):
            paper_info["paper_abstract"] = source_abstract
            abstract_source = paper_info.get("abstract_source") or "source_metadata"
            extraction_status = "success"
            counts["source_abstract"] += 1
        elif extracted_abstract:
            paper_info["paper_abstract"] = extracted_abstract
            abstract_source = "generated_from_pdf" if generated else "pdf_front_pages"
            extraction_status = "success"
            counts["extracted_abstract"] += 1
        else:
            paper_info["paper_abstract"] = None
            abstract_source = "missing"
            extraction_status = "failed" if content_error else "not_found"
            counts["missing_abstract"] += 1
        paper_info["abstract_source"] = abstract_source

        if source_keywords and not initial_pdf_text:
            paper_info["keywords"] = _keywords_text(source_keywords)
            keyword_source = "source_metadata"
            keyword_extraction_status = "fallback"
            counts["source_keyword"] += 1
        elif extracted_keywords:
            paper_info["keywords"] = _keywords_text(extracted_keywords)
            keyword_source = "pdf_front_pages"
            keyword_extraction_status = "success"
            counts["extracted_keyword"] += 1
        elif source_keywords:
            paper_info["keywords"] = _keywords_text(source_keywords)
            keyword_source = "source_metadata"
            keyword_extraction_status = "fallback"
            counts["source_keyword"] += 1
        else:
            paper_info["keywords"] = None
            keyword_source = "missing"
            keyword_extraction_status = "failed" if content_error else "not_found"
            counts["missing_keyword"] += 1

        paper["abstract_resolution"] = {
            "source": abstract_source,
            "extraction_status": extraction_status,
            "pdf_pages_read": page_count,
            "ai_summary_source": (
                ("pdf_front_pages" if initial_pdf_text else "source_metadata")
                if content_error is None else "failed"
            ),
            "error": content_error,
            "pdf_error": pdf_error,
        }
        paper["keyword_resolution"] = {
            "source": keyword_source,
            "extraction_status": keyword_extraction_status,
            "pdf_pages_read": page_count,
            "error": content_error,
        }
        warning = (
            f"{paper_info.get('title')} 的摘要补全警告：{content_error or pdf_error}"
            if content_error or pdf_error
            else None
        )
        return counts, warning

    @staticmethod
    async def _gather_or_cancel(
        tasks: list[asyncio.Task[tuple[dict[str, int], str | None]]],
    ) -> list[tuple[dict[str, int], str | None]]:
        try:
            return list(await asyncio.gather(*tasks))
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            child_run_id = state["child_run_id"]
            artifact_uri = state.get("pdf_manifest_artifact_ref")

            manifest = await self.artifact_store.read_json_uri(artifact_uri)
            papers = manifest.get("papers", [])

            counts = {
                "source_abstract": 0,
                "extracted_abstract": 0,
                "missing_abstract": 0,
                "front_pages_read": 0,
                "ai_abstract": 0,
                "source_keyword": 0,
                "extracted_keyword": 0,
                "missing_keyword": 0,
            }
            warnings: list[str] = []

            semaphore = asyncio.Semaphore(ABSTRACT_ENRICHMENT_CONCURRENCY)
            tasks = [
                asyncio.create_task(self._enrich_paper(paper, semaphore))
                for paper in papers
            ]
            for paper_counts, warning in await self._gather_or_cancel(tasks):
                for key, value in paper_counts.items():
                    counts[key] += value
                if warning:
                    warnings.append(warning)

            manifest["papers"] = papers
            manifest["step_key"] = "abstract_enrichment"

            artifact = await self.artifact_store.write_json(
                run_id = child_run_id,
                step_key = "abstract_enrichment",
                source = "abstract",
                kind = "paper_info_abstract_manifest_json",
                payload = manifest,
                count = len(papers),
                metadata = {
                    "input_manifest": artifact_uri,
                    **{f"{key}_count": value for key, value in counts.items()},
                },
            )

        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"摘要补充失败：{exc}",
            }

        return {
            "stage": "recommending",
            "status": "running",
            "abstract_manifest_artifact_ref": artifact.artifact_uri,
            "progress": {
                **state.get("progress", {}),
                "abstract_from_source": counts["source_abstract"],
                "abstract_extracted_from_pdf": (
                    counts["extracted_abstract"]
                ),
                "abstract_missing": counts["missing_abstract"],
                "pdf_front_pages_read": counts["front_pages_read"],
                "ai_abstract_generated": counts["ai_abstract"],
                "keywords_from_source": counts["source_keyword"],
                "keywords_extracted_from_pdf": counts["extracted_keyword"],
                "keywords_missing": counts["missing_keyword"],
            },
            "warnings": [
                *state.get("warnings", []),
                *warnings,
            ],
            "error": None,
        }
