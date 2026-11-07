from __future__ import annotations

from typing import Any

from langchain.tools import BaseTool
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel, Field

from app.rag.retrieval.paper_content_retrieval import PaperContentHybridRetrievalModule

MAX_QUERIES = 3
MAX_QUERY_LENGTH = 160
QUERY_TOP_K = 4
MAX_CHUNK_CHARS = 3_000
MAX_RESULT_CHARS = 12_000


class SearchPaperIndexArgs(BaseModel):
    querys: list[str] = Field(
        min_length=1,
        max_length=MAX_QUERIES,
        description="一到三条当前论文内容检索语句。",
    )


def _clip(value: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    return f"{value[: max_chars - 1]}…" if max_chars > 1 else "…"


async def search_paper_index(
    params: dict[str, Any],
    *,
    paper_id: str,
    user_id: str,
    content_retrieval: PaperContentHybridRetrievalModule,
    top_k: int = QUERY_TOP_K,
) -> list[dict[str, Any]]:
    args = SearchPaperIndexArgs.model_validate(params)
    remaining = MAX_RESULT_CHARS
    results = []
    for query in args.querys:
        documents = await content_retrieval.search(
            query,
            user_id=user_id,
            top_k=top_k,
            paper_ids=[paper_id],
        )
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

    return results


def build_search_paper_index_tool(
    *,
    content_retrieval: PaperContentHybridRetrievalModule,
    top_k: int = QUERY_TOP_K,
) -> BaseTool:
    @tool(
        "search_paper_index",
        description=(
            "查找论文的方法、实验、结论或支持论点的原文证据时，调用此工具。"
            "输入querys 中的一到三条具体内容检索语句。"
        ),
        args_schema=SearchPaperIndexArgs,
    )
    async def search_paper_index_tool(
        querys: list[str],
        runtime: ToolRuntime[Any],
    ) -> list[dict[str, Any]]:
        return await search_paper_index(
            {"querys": querys},
            paper_id=str(runtime.context.paper_id),
            content_retrieval=content_retrieval,
            user_id=str(runtime.context.user_id),
            top_k=top_k,
        )

    return search_paper_index_tool
