from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from langchain_core.tools import BaseTool, tool
from app.rag.retrieval.paper_content_retrieval import PaperContentHybridRetrievalModule


MAX_QUERIES = 3
MAX_QUERY_LENGTH = 160
QUERY_TOP_K = 4
MAX_CHUNK_CHARS = 3_000
MAX_RESULT_CHARS = 12_000


class SearchPaperIndexArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    querys: list[str] = Field(
        min_length=1,
        max_length=MAX_QUERIES,
        description="一到三条当前论文内容检索语句。",
    )
    paper_id: str = Field(
        min_length=1,
        description="要检索的目标论文 ID。",
    )

    @field_validator("querys")
    @classmethod
    def normalize_queries(cls, value: list[str]) -> list[str]:
        queries = []
        for query in value:
            normalized = " ".join(query.split())
            if not normalized:
                raise ValueError("querys must not contain blank queries")
            if len(normalized) > MAX_QUERY_LENGTH:
                raise ValueError(f"each query must be at most {MAX_QUERY_LENGTH} characters")
            queries.append(normalized)
        return queries

    @field_validator("paper_id")
    @classmethod
    def normalize_paper_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("paper_id must not be blank")
        return value


def _clip(value: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars - 1]}…" if max_chars > 1 else "…"


async def search_paper_index(
    params: dict[str, Any],
    *,
    content_retrieval: PaperContentHybridRetrievalModule,
    top_k: int = QUERY_TOP_K,
) -> dict[str, Any]:
    args = SearchPaperIndexArgs.model_validate(params)
    result_sets = await asyncio.gather(
        *(
            content_retrieval.search(
                query,
                top_k=top_k,
                paper_ids=[args.paper_id],
            )
            for query in args.querys
        )
    )

    remaining = MAX_RESULT_CHARS
    results = []
    for query, documents in zip(args.querys, result_sets, strict=True):
        chunks = []
        for document in documents:
            text = str(document.page_content or "").strip()
            if not text or remaining <= 0:
                break
            text = _clip(text, min(MAX_CHUNK_CHARS, remaining))
            remaining -= len(text)
            chunks.append(
                {
                    "chunk_id": str(document.metadata["chunk_id"]),
                    "section_path": str(document.metadata.get("section_path") or ""),
                    "page_numbers": list(document.metadata.get("page_numbers") or []),
                    "text": text,
                }
            )
        results.append({"query": query, "chunks": chunks})

    return {"paper_id": args.paper_id, "results": results}


def build_search_paper_index_tool(
    *,
    content_retrieval: PaperContentHybridRetrievalModule,
    top_k: int = QUERY_TOP_K,
) -> BaseTool:
    @tool(
        "search_paper_index",
        description=(
            "在指定论文的已有索引中检索相关 Markdown 正文片段。"
            "当需要查找某篇论文的方法、实验、结论或支持论点的原文证据时，调用此工具。"
            "输入应包含目标论文 paper_id，以及 querys 中的一到三条具体内容检索语句。"
            "检索范围仅限该论文，不用于外部论文搜索。"
        ),
        args_schema=SearchPaperIndexArgs,
    )
    async def search_paper_index_tool(querys: list[str], paper_id: str) -> dict[str, Any]:
        """在指定论文的索引内容中批量检索 Markdown 片段。"""
        return await search_paper_index(
            {"querys": querys, "paper_id": paper_id},
            content_retrieval=content_retrieval,
            top_k=top_k,
        )

    return search_paper_index_tool
