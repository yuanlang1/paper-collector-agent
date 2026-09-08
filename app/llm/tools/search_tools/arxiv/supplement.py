from __future__ import annotations

import difflib
import re
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field

from app.llm.tools.registry import Tool, ToolExecutionContext
from app.llm.tools.search_tools.arxiv.search_arxiv import arxiv_search_handler


class ArxivSupplementArgs(BaseModel):
    papers: list[dict[str, Any]] = Field(...)
    fill_fields: list[Literal[
        "summary",
        "pdf_url",
        "abstract_url",
        "arxiv_id",
        "published",
        "updated",
        "authors",
        "categories",
        "doi",
    ]] = Field(
        default_factory=lambda: [
            "summary",
            "pdf_url",
            "abstract_url",
            "arxiv_id",
        ]
    )
    match_threshold: float = 0.88
    max_candidates: int = 3
    only_when_missing: bool = True


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _normalize_title(title: str | None) -> str:
    title = _clean_text(title).lower()

    # 去掉末尾标点
    title = re.sub(r"[。.!?：:；;,，]+$", "", title)

    # 去掉多余符号，仅保留常见英文、数字、空格
    title = re.sub(r"[^a-z0-9\s\-]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()

    return title


def _normalize_doi(doi: str | None) -> str:
    if not doi:
        return ""

    doi = doi.strip()
    doi = doi.replace("https://doi.org/", "")
    doi = doi.replace("http://doi.org/", "")
    doi = doi.replace("doi:", "")
    doi = doi.strip().lower()

    return doi


def _title_similarity(a: str | None, b: str | None) -> float:
    na = _normalize_title(a)
    nb = _normalize_title(b)

    if not na or not nb:
        return 0.0

    return difflib.SequenceMatcher(None, na, nb).ratio()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, str) and not value.strip():
        return True

    if isinstance(value, list) and len(value) == 0:
        return True

    return False


def _need_supplement(paper: dict[str, Any], fill_fields: list[str]) -> bool:
    for field in fill_fields:
        if _is_missing(paper.get(field)):
            return True

    return False


def _merge_arxiv_fields(
    *,
    original: dict[str, Any],
    arxiv_paper: dict[str, Any],
    fill_fields: list[str],
    only_when_missing: bool,
) -> tuple[dict[str, Any], list[str]]:
    """
    将 arXiv 字段补充到原 paper 中。
    返回：
    - 合并后的 paper
    - 实际补充的字段列表
    """

    merged = dict(original)
    filled_fields: list[str] = []

    field_mapping = {
        "summary": "summary",
        "pdf_url": "pdf_url",
        "abstract_url": "abstract_url",
        "arxiv_id": "arxiv_id",
        "published": "published",
        "updated": "updated",
        "authors": "authors",
        "categories": "categories",
        "doi": "doi",
    }

    for target_field in fill_fields:
        source_field = field_mapping.get(target_field)
        if not source_field:
            continue

        source_value = arxiv_paper.get(source_field)

        if _is_missing(source_value):
            continue

        if only_when_missing:
            if _is_missing(merged.get(target_field)):
                merged[target_field] = source_value
                filled_fields.append(target_field)
        else:
            merged[target_field] = source_value
            filled_fields.append(target_field)

    # 额外保留补全来源，方便后续调试和入库
    if filled_fields:
        merged["enriched_by"] = "ARXIV"
        merged["arxiv_match_title"] = arxiv_paper.get("title")
        merged["arxiv_abstract_url"] = arxiv_paper.get("abstract_url")
        merged["arxiv_pdf_url"] = arxiv_paper.get("pdf_url")
        merged["arxiv_id"] = arxiv_paper.get("arxiv_id") or merged.get("arxiv_id")

    return merged, filled_fields


async def _search_arxiv_by_doi(
    *,
    doi: str,
    max_candidates: int,
    context: ToolExecutionContext,
) -> list[dict[str, Any]]:
    """
    用 DOI 搜 arXiv。
    注意：
    - arXiv 未提供 DOI 专用检索字段，因此使用 DOI 作为主题查询词。
    - 部分 DOI 在 arXiv 中可能不存在，所以必须允许失败和空结果。
    """

    normalized_doi = _normalize_doi(doi)

    if not normalized_doi:
        return []

    # 优先尝试高级查询
    result = await arxiv_search_handler(
        {
            "query": normalized_doi,
            "search_type": "topic",
            "max_results": max_candidates,
            "total_limit": max_candidates,
            "sort": "relevance",
            "include_abstract": True,
        },
        context,
    )

    if result.get("ok") and result.get("papers"):
        return result["papers"]

    # 兜底：把 DOI 当普通关键词搜
    fallback = await arxiv_search_handler(
        {
            "query": normalized_doi,
            "search_type": "topic",
            "max_results": max_candidates,
            "total_limit": max_candidates,
            "sort": "relevance",
            "include_abstract": True,
        },
        context,
    )

    if fallback.get("ok") and fallback.get("papers"):
        return fallback["papers"]

    return []


async def _search_arxiv_by_title(
    *,
    title: str,
    max_candidates: int,
    context: ToolExecutionContext,
) -> list[dict[str, Any]]:
    title = _clean_text(title)

    if not title:
        return []

    # 先标题字段精确/短语搜索
    result = await arxiv_search_handler(
        {
            "query": title,
            "search_type": "title",
            "max_results": max_candidates,
            "total_limit": max_candidates,
            "sort": "relevance",
            "include_abstract": True,
        },
        context,
    )

    if result.get("ok") and result.get("papers"):
        return result["papers"]

    # 兜底：全字段搜索
    fallback = await arxiv_search_handler(
        {
            "query": title,
            "search_type": "topic",
            "max_results": max_candidates,
            "total_limit": max_candidates,
            "sort": "relevance",
            "include_abstract": True,
        },
        context,
    )

    if fallback.get("ok") and fallback.get("papers"):
        return fallback["papers"]

    return []


def _pick_best_candidate(
    *,
    original: dict[str, Any],
    candidates: list[dict[str, Any]],
    match_threshold: float,
) -> tuple[dict[str, Any] | None, float, str | None]:
    """
    从 arXiv 候选中选择最佳匹配。
    匹配策略：
    1. DOI 完全一致，直接命中
    2. 标题相似度超过阈值，选择最高的
    """

    if not candidates:
        return None, 0.0, None

    original_doi = _normalize_doi(original.get("doi"))
    original_title = original.get("title")

    # 1. DOI 精确匹配
    if original_doi:
        for candidate in candidates:
            candidate_doi = _normalize_doi(candidate.get("doi"))
            if candidate_doi and candidate_doi == original_doi:
                return candidate, 1.0, "doi"

    # 2. 标题模糊匹配
    best_candidate = None
    best_score = 0.0

    for candidate in candidates:
        score = _title_similarity(original_title, candidate.get("title"))

        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate and best_score >= match_threshold:
        return best_candidate, best_score, "title"

    return None, best_score, None


async def arxiv_supplement_handler(
    params: Dict[str, Any],
    context: ToolExecutionContext,
) -> Dict[str, Any]:
    args = ArxivSupplementArgs(**params)

    enriched_papers: list[dict[str, Any]] = []
    supplement_records: list[dict[str, Any]] = []

    total_checked = 0
    total_enriched = 0
    total_not_found = 0
    total_skipped = 0

    for paper in args.papers:
        total_checked += 1

        if not _need_supplement(paper, args.fill_fields):
            enriched_papers.append(paper)
            total_skipped += 1
            continue

        candidates: list[dict[str, Any]] = []

        # 1. 优先 DOI
        doi = paper.get("doi")
        if doi:
            candidates = await _search_arxiv_by_doi(
                doi=doi,
                max_candidates=args.max_candidates,
                context=context,
            )

        # 2. DOI 未命中，再用 title
        if not candidates:
            title = paper.get("title")
            candidates = await _search_arxiv_by_title(
                title=title,
                max_candidates=args.max_candidates,
                context=context,
            )

        best_candidate, score, match_by = _pick_best_candidate(
            original=paper,
            candidates=candidates,
            match_threshold=args.match_threshold,
        )

        if not best_candidate:
            enriched_papers.append(paper)
            total_not_found += 1

            supplement_records.append(
                {
                    "title": paper.get("title"),
                    "doi": paper.get("doi"),
                    "matched": False,
                    "best_score": score,
                    "candidate_count": len(candidates),
                }
            )
            continue

        merged, filled_fields = _merge_arxiv_fields(
            original=paper,
            arxiv_paper=best_candidate,
            fill_fields=args.fill_fields,
            only_when_missing=args.only_when_missing,
        )

        enriched_papers.append(merged)

        if filled_fields:
            total_enriched += 1

        supplement_records.append(
            {
                "title": paper.get("title"),
                "doi": paper.get("doi"),
                "matched": True,
                "match_by": match_by,
                "match_score": score,
                "filled_fields": filled_fields,
                "arxiv_id": best_candidate.get("arxiv_id"),
                "arxiv_title": best_candidate.get("title"),
                "arxiv_pdf_url": best_candidate.get("pdf_url"),
                "arxiv_abstract_url": best_candidate.get("abstract_url"),
            }
        )

    return {
        "ok": True,
        "source": "ARXIV_SUPPLEMENT",
        "checked_count": total_checked,
        "enriched_count": total_enriched,
        "skipped_count": total_skipped,
        "not_found_count": total_not_found,
        "papers": enriched_papers,
        "supplement_records": supplement_records,
    }


ARXIV_SUPPLEMENT_TOOL = Tool(
    name="arxiv_supplement",
    description=(
        "当 DBLP 或 Google Scholar 检索结果中的论文缺少 summary、pdf_url、"
        "abstract_url 或 arxiv_id 时，使用 DOI 或标题到 arXiv 检索并补全字段。"
        "该工具不负责初始论文检索，只负责对已有 papers 做字段增强。"
    ),
    input_schema=ArxivSupplementArgs.model_json_schema(),
    fn=arxiv_supplement_handler,
)
