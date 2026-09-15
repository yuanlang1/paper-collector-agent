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
from app.llm.model_factory import create_validated_structured_chat_model
from pypdf import PdfReader


PDF_READ_PAGE_COUNT = 4
MAX_ABSTRACT_CHARS = 8_000
MAX_KEYWORDS = 12
MAX_KEYWORD_CHARS = 1_000


PDF_FRONT_PAGE_ENRICHMENT_SYSTEM_PROMPT = """
你负责仅根据论文前几页文本完成三个字段。

- paper_abstract：提取页面中原始的 Abstract 段落，保留原文语言和含义；
  未找到时返回 null，不得改写或编造。
- keywords：提取页面中 Keywords、Index Terms 或“关键词”段落；如果未找到就根据论文内容生成关键词。
- ai_abstract：使用中文概括研究问题、核心方法、实验设置、主要结果和贡献；
  仅依据页面文本，不得编造其中没有的事实。
""".strip()


class PdfFrontPageEnrichmentResult(BaseModel):
    paper_abstract: str | None = Field(
        default=None,
        max_length=MAX_ABSTRACT_CHARS,
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
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.model = model or create_validated_structured_chat_model(
            PdfFrontPageEnrichmentResult,
            temperature=0,
        )

    async def _enrich_from_pdf_front_pages(
        self,
        initial_pdf_text: str,
    ) -> PdfFrontPageEnrichmentResult:
        result = await self.model.ainvoke(
            [
                SystemMessage(content=PDF_FRONT_PAGE_ENRICHMENT_SYSTEM_PROMPT),
                HumanMessage(
                    content=json.dumps(
                        {"pdf_front_pages": initial_pdf_text},
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

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            run_id = state.get("run_id")
            artifact_uri = state.get("pdf_manifest_artifact_ref")

            if not isinstance(run_id, str) or not run_id:
                raise ValueError("缺少有效 run_id。")

            if (
                not isinstance(artifact_uri, str)
                or not artifact_uri.startswith("artifact://")
            ):
                raise ValueError("缺少 PDF manifest artifact。")

            base_dir = self.artifact_store.base_dir.resolve()
            manifest_path = (
                base_dir
                / artifact_uri.removeprefix("artifact://")
            ).resolve()

            try:
                manifest_path.relative_to(base_dir)
            except ValueError as exc:
                raise ValueError(
                    "abstract manifest 超出 artifact 存储目录。"
                ) from exc

            def read_manifest() -> dict[str, Any]:
                with manifest_path.open(
                    "r",
                    encoding = "utf-8",
                ) as file:
                    return json.load(file)

            manifest = await asyncio.to_thread(read_manifest)
            papers = manifest.get("papers", [])

            if not isinstance(papers, list):
                raise ValueError("manifest papers 格式无效。")

            source_abstract_count = 0
            extracted_abstract_count = 0
            missing_abstract_count = 0
            front_pages_read_count = 0
            ai_abstract_count = 0
            source_keyword_count = 0
            extracted_keyword_count = 0
            missing_keyword_count = 0
            warnings: list[str] = []

            for paper in papers:
                if not isinstance(paper, dict):
                    continue

                paper_info = paper.get("paper_info")
                pdf_download = paper.get("pdf_download") or {}

                if not isinstance(paper_info, dict):
                    continue

                source_abstract = _text(
                    paper_info.get("paper_abstract")
                )
                source_keywords = _normalize_keywords(
                    paper_info.get("keywords")
                )
                local_pdf_path = _text(
                    pdf_download.get("local_pdf_path")
                )
                page_count = 0
                content_error: str | None = None
                extracted_abstract: str | None = None
                extracted_keywords: list[str] = []

                try:
                    if not local_pdf_path:
                        raise ValueError("缺少本地 PDF 路径。")

                    initial_pdf_text, page_count = (
                        await asyncio.to_thread(
                            _extract_initial_pdf_text,
                            Path(local_pdf_path),
                        )
                    )
                    front_pages_read_count += 1
                    enrichment = await self._enrich_from_pdf_front_pages(
                        initial_pdf_text
                    )
                    extracted_abstract = _text(enrichment.paper_abstract)
                    extracted_keywords = _normalize_keywords(enrichment.keywords)
                    ai_abstract = _text(enrichment.ai_abstract)
                    if ai_abstract is None:
                        raise ValueError("AI 摘要为空。")
                    paper_info["ai_abstract"] = ai_abstract
                    ai_abstract_count += 1
                except Exception as exc:
                    content_error = str(exc)
                    paper_info["ai_abstract"] = None
                    warnings.append(
                        f"{paper_info.get('title')} 的 PDF 内容补全失败：{exc}"
                    )

                if extracted_abstract:
                    paper_info["paper_abstract"] = extracted_abstract
                    abstract_source = "pdf_front_pages"
                    extraction_status = "success"
                    extracted_abstract_count += 1
                elif source_abstract:
                    paper_info["paper_abstract"] = source_abstract
                    abstract_source = "source_metadata"
                    extraction_status = "fallback"
                    source_abstract_count += 1
                else:
                    paper_info["paper_abstract"] = None
                    abstract_source = "missing"
                    extraction_status = "failed" if content_error else "not_found"
                    missing_abstract_count += 1

                if extracted_keywords:
                    paper_info["keywords"] = _keywords_text(extracted_keywords)
                    keyword_source = "pdf_front_pages"
                    keyword_extraction_status = "success"
                    extracted_keyword_count += 1
                elif source_keywords:
                    paper_info["keywords"] = _keywords_text(source_keywords)
                    keyword_source = "source_metadata"
                    keyword_extraction_status = "fallback"
                    source_keyword_count += 1
                else:
                    paper_info["keywords"] = None
                    keyword_source = "missing"
                    keyword_extraction_status = (
                        "failed" if content_error else "not_found"
                    )
                    missing_keyword_count += 1

                paper["abstract_resolution"] = {
                    "source": abstract_source,
                    "extraction_status": extraction_status,
                    "pdf_pages_read": page_count,
                    "ai_summary_source": (
                        "pdf_front_pages" if content_error is None else "failed"
                    ),
                    "error": content_error,
                }
                paper["keyword_resolution"] = {
                    "source": keyword_source,
                    "extraction_status": keyword_extraction_status,
                    "pdf_pages_read": page_count,
                    "error": content_error,
                }

            manifest["papers"] = papers
            manifest["step_key"] = "abstract_enrichment"

            artifact = await self.artifact_store.write_json(
                run_id = run_id,
                step_key = "abstract_enrichment",
                source = "abstract",
                kind = "paper_info_abstract_manifest_json",
                payload = manifest,
                count = len(papers),
                metadata = {
                    "input_manifest": artifact_uri,
                    "source_abstract_count": source_abstract_count,
                    "extracted_abstract_count": (
                        extracted_abstract_count
                    ),
                    "missing_abstract_count": missing_abstract_count,
                    "front_pages_read_count": front_pages_read_count,
                    "ai_abstract_count": ai_abstract_count,
                    "source_keyword_count": source_keyword_count,
                    "extracted_keyword_count": extracted_keyword_count,
                    "missing_keyword_count": missing_keyword_count,
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
            "status": (
                "partial_failed"
                if state.get("degraded") or warnings
                else "running"
            ),
            "degraded": bool(state.get("degraded")) or bool(warnings),
            "abstract_manifest_artifact_ref": artifact.artifact_uri,
            "progress": {
                **state.get("progress", {}),
                "abstract_from_source": source_abstract_count,
                "abstract_extracted_from_pdf": (
                    extracted_abstract_count
                ),
                "abstract_missing": missing_abstract_count,
                "pdf_front_pages_read": front_pages_read_count,
                "ai_abstract_generated": ai_abstract_count,
                "keywords_from_source": source_keyword_count,
                "keywords_extracted_from_pdf": extracted_keyword_count,
                "keywords_missing": missing_keyword_count,
            },
            "warnings": [
                *state.get("warnings", []),
                *warnings,
            ],
            "error": None,
        }
