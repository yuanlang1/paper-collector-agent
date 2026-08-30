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


AI_ABSTRACT_SYSTEM_PROMPT = """
    你负责根据论文前几页内容生成中文 AI 摘要。

    规则：
    - 使用中文；
    - 概括研究问题、核心方法、实验设置、主要结果和贡献；
    - 可参考输入的原始摘要；
    - 不得编造输入中没有的实验结果、指标、方法细节或结论；
    - 输出适合作为论文检索系统展示的简洁摘要。
""".strip()


FALLBACK_ABSTRACT_SYSTEM_PROMPT = """
    你负责基于有限论文信息生成谨慎的中文 AI 摘要。

    若有原始摘要，仅根据原始摘要生成中文概括。
    若没有原始摘要，只能依据标题、作者、关键词和来源描述研究方向，
    不得编造实验、方法细节、结果、结论或指标。
""".strip()


KEYWORD_GENERATION_SYSTEM_PROMPT = """
    根据论文标题、原始摘要和前几页正文生成论文关键词。

    规则：
    - 仅使用输入中有证据支持的术语，不得虚构方法、数据集、指标或结论；
    - 生成 3 到 8 个简洁的学术短语，而不是完整句子；
    - 优先保持论文原文语言；英文论文使用英文术语；
    - 不要输出“Keywords”“Index Terms”等标题文本；
    - 不要重复、不要编号、不要包含解释。
""".strip()


class AiAbstractResult(BaseModel):
    ai_abstract: str = Field(min_length = 1, max_length = 3_000)


class KeywordGenerationResult(BaseModel):
    keywords: list[str] = Field(min_length=3, max_length=8)


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


def _extract_original_abstract(initial_pdf_text: str) -> str:
    match = re.search(
        r"""
        (?is)
        \babstract\b
        \s*[:\-—]?\s*
        (?P<abstract>.+?)
        (? = 
            \n\s*(?:keywords?|index\s+terms?)\b
            |\n\s*(?:\d+|[ivxlcdm]+)\.?\s*introduction\b
            |\n\s*introduction\b
        )
        """,
        initial_pdf_text,
        flags = re.VERBOSE,
    )

    if match is None:
        raise ValueError("未在 PDF 前四页定位到 Abstract 段落。")

    abstract = re.sub(
        r"\s+",
        " ",
        match.group("abstract"),
    ).strip()

    if len(abstract) < 80:
        raise ValueError("提取到的 Abstract 内容过短。")

    return abstract[:MAX_ABSTRACT_CHARS]


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


def _extract_original_keywords(initial_pdf_text: str) -> list[str]:
    match = re.search(
        r"""
        (?is)
        (?:^|\n)\s*(?:keywords?|key\s*words?|index\s+terms?|关键词)
        \s*[:：\-—]?\s*
        (?P<keywords>.+?)
        (?=
            \n\s*(?:\d+|[ivxlcdm]+)\.?\s*(?:introduction|引言)\b
            |\n\s*(?:introduction|引言)\b
            |\n{2,}
            |\Z
        )
        """,
        initial_pdf_text,
        flags=re.VERBOSE,
    )
    if match is None:
        return []

    return _normalize_keywords(match.group("keywords"))


class AbstractEnrichNode:
    def __init__(
        self,
        artifact_store: LocalArtifactStore | None = None,
        model: Any | None = None,
        keyword_model: Any | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.model = model or create_validated_structured_chat_model(
            AiAbstractResult,
            temperature = 0,
        )
        self.keyword_model = keyword_model or create_validated_structured_chat_model(
            KeywordGenerationResult,
            temperature=0,
        )

    async def _generate_ai_abstract(
        self,
        *,
        paper_info: Mapping[str, Any],
        initial_pdf_text: str,
    ) -> str:
        result = await self.model.ainvoke(
            [
                SystemMessage(
                    content = AI_ABSTRACT_SYSTEM_PROMPT,
                ),
                HumanMessage(
                    content = json.dumps(
                        {
                            "title": paper_info.get("title"),
                            "authors": paper_info.get("authors"),
                            "paper_abstract": paper_info.get(
                                "paper_abstract"
                            ),
                            "pdf_front_pages": initial_pdf_text,
                        },
                        ensure_ascii = False,
                    )
                ),
            ]
        )

        output = (
            result
            if isinstance(result, AiAbstractResult)
            else AiAbstractResult.model_validate(result)
        )

        return output.ai_abstract.strip()

    async def _generate_fallback_ai_abstract(
        self,
        paper_info: Mapping[str, Any],
    ) -> str:
        result = await self.model.ainvoke(
            [
                SystemMessage(
                    content = FALLBACK_ABSTRACT_SYSTEM_PROMPT,
                ),
                HumanMessage(
                    content = json.dumps(
                        {
                            "title": paper_info.get("title"),
                            "authors": paper_info.get("authors"),
                            "keywords": paper_info.get("keywords"),
                            "source": paper_info.get("source"),
                            "paper_abstract": paper_info.get(
                                "paper_abstract"
                            ),
                        },
                        ensure_ascii = False,
                    )
                ),
            ]
        )

        output = (
            result
            if isinstance(result, AiAbstractResult)
            else AiAbstractResult.model_validate(result)
        )

        return output.ai_abstract.strip()

    async def _generate_keywords(
        self,
        *,
        paper_info: Mapping[str, Any],
        initial_pdf_text: str | None,
    ) -> list[str]:
        result = await self.keyword_model.ainvoke(
            [
                SystemMessage(content=KEYWORD_GENERATION_SYSTEM_PROMPT),
                HumanMessage(
                    content=json.dumps(
                        {
                            "title": paper_info.get("title"),
                            "paper_abstract": paper_info.get(
                                "paper_abstract"
                            ),
                            "pdf_front_pages": initial_pdf_text,
                        },
                        ensure_ascii=False,
                    )
                ),
            ]
        )
        output = (
            result
            if isinstance(result, KeywordGenerationResult)
            else KeywordGenerationResult.model_validate(result)
        )
        keywords = _normalize_keywords(output.keywords)
        if len(keywords) < 3:
            raise ValueError("generated fewer than three valid keywords")
        return keywords

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
            generated_keyword_count = 0
            missing_keyword_count = 0
            warnings: list[str] = []

            for paper in papers:
                if not isinstance(paper, dict):
                    continue

                paper_info = paper.get("paper_info")
                pdf_download = paper.get("pdf_download") or {}

                if not isinstance(paper_info, dict):
                    continue

                original_abstract = _text(
                    paper_info.get("paper_abstract")
                )
                local_pdf_path = _text(
                    pdf_download.get("local_pdf_path")
                )

                initial_pdf_text: str | None = None
                page_count = 0
                extraction_error: str | None = None

                if original_abstract:
                    abstract_source = "source"
                    extraction_status = "not_needed"
                    source_abstract_count += 1
                else:
                    abstract_source = "missing"
                    extraction_status = "pending"

                source_keywords = _normalize_keywords(
                    paper_info.get("keywords")
                )
                keyword_source = "missing"
                keyword_extraction_status = "pending"
                keyword_error: str | None = None

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

                    if not original_abstract:
                        try:
                            paper_info["paper_abstract"] = (
                                _extract_original_abstract(
                                    initial_pdf_text
                                )
                            )
                            abstract_source = "pdf_extracted"
                            extraction_status = "success"
                            extracted_abstract_count += 1
                        except Exception as exc:
                            abstract_source = "pdf_abstract_not_found"
                            extraction_status = "failed"
                            extraction_error = str(exc)
                            missing_abstract_count += 1

                except Exception as exc:
                    extraction_error = str(exc)

                    if not original_abstract:
                        abstract_source = "metadata_fallback"
                        extraction_status = "failed"
                        missing_abstract_count += 1
                    else:
                        extraction_status = "front_pages_read_failed"

                if source_keywords:
                    paper_info["keywords"] = _keywords_text(source_keywords)
                    keyword_source = "source_metadata"
                    keyword_extraction_status = "not_needed"
                    source_keyword_count += 1
                elif initial_pdf_text:
                    extracted_keywords = _extract_original_keywords(
                        initial_pdf_text
                    )
                    if extracted_keywords:
                        paper_info["keywords"] = _keywords_text(
                            extracted_keywords
                        )
                        keyword_source = "pdf_extracted"
                        keyword_extraction_status = "success"
                        extracted_keyword_count += 1
                    else:
                        keyword_extraction_status = "not_found"
                        keyword_error = (
                            "keywords section not found in PDF front pages"
                        )

                if keyword_source == "missing":
                    try:
                        generated_keywords = await self._generate_keywords(
                            paper_info=paper_info,
                            initial_pdf_text=initial_pdf_text,
                        )
                        paper_info["keywords"] = _keywords_text(
                            generated_keywords
                        )
                        keyword_source = "generated_from_paper"
                        keyword_extraction_status = "generated"
                        keyword_error = None
                        generated_keyword_count += 1
                    except Exception as exc:
                        paper_info["keywords"] = None
                        keyword_extraction_status = "failed"
                        keyword_error = str(exc)
                        missing_keyword_count += 1
                        warnings.append(
                            f"{paper_info.get('title')} 的论文关键词生成失败：{exc}"
                        )

                try:
                    if initial_pdf_text:
                        paper_info["ai_abstract"] = (
                            await self._generate_ai_abstract(
                                paper_info = paper_info,
                                initial_pdf_text = initial_pdf_text,
                            )
                        )
                        ai_summary_source = "pdf_front_pages"
                    else:
                        paper_info["ai_abstract"] = (
                            await self._generate_fallback_ai_abstract(
                                paper_info
                            )
                        )
                        ai_summary_source = (
                            "source_abstract"
                            if paper_info.get("paper_abstract")
                            else "metadata"
                        )

                    ai_abstract_count += 1

                except Exception as exc:
                    paper_info["ai_abstract"] = None
                    ai_summary_source = "failed"
                    warnings.append(
                        f"{paper_info.get('title')} 的中文 AI 摘要生成失败："
                        f"{exc}"
                    )

                paper["abstract_resolution"] = {
                    "source": abstract_source,
                    "extraction_status": extraction_status,
                    "pdf_pages_read": page_count,
                    "ai_summary_source": ai_summary_source,
                    "error": extraction_error,
                }
                paper["keyword_resolution"] = {
                    "source": keyword_source,
                    "extraction_status": keyword_extraction_status,
                    "pdf_pages_read": page_count,
                    "error": keyword_error,
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
                    "generated_keyword_count": generated_keyword_count,
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
                "keywords_generated": generated_keyword_count,
                "keywords_missing": missing_keyword_count,
            },
            "warnings": [
                *state.get("warnings", []),
                *warnings,
            ],
            "error": None,
        }
