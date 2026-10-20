from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from app.infrastructure.grpc.paper_service_grpc_client import (
    paper_service_grpc_client,
)
from langchain_core.tools import ToolException, tool


class GetTaskPapersArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: StrictInt = Field(
        ..., gt=0, description="要查看论文的检索任务 ID。"
    )


async def get_task_papers_handler(
    params: dict[str, Any],
    context: object | None = None,
) -> dict[str, Any]:
    del context
    request = GetTaskPapersArgs.model_validate(params)
    result = await paper_service_grpc_client.get_task_review_papers(
        request.task_id
    )
    if not result.get("ok"):
        raise ToolException(str(result.get("error") or "任务论文查询失败。"))

    papers = result.get("result", {}).get("papers", [])
    payload = {
        "papers": [
            {
                "title": paper.get("title"),
                "authors": paper.get("authors"),
                "abstract": paper.get("paper_abstract"),
                "pdf_url": paper.get("pdf_url"),
            }
            for paper in papers
        ]
    }
    return payload


@tool(
    "get_task_papers",
    description=(
        "获取指定检索任务下的全部论文，返回标题、作者、摘要和 PDF 地址。"
        "当用户想了解任务中主要有哪些论文、涉及哪些研究主题，或基于任务论文推荐综述题目和研究方向时，调用此工具。"
        "输入应为已有检索任务的正整数 task_id，不要求 RAG 索引完成。"
    ),
    args_schema=GetTaskPapersArgs,
)
async def get_task_papers_tool(
    *,
    task_id: StrictInt,
) -> dict[str, Any]:
    """获取指定检索任务下的全部论文，返回标题、作者、摘要和 PDF 地址。用于查看这个任务主要有哪些论文、了解论文涉及的研究主题，以及依据任务中的论文推荐综述题目或研究方向。不要求 RAG 索引完成。"""
    params = {
        "task_id": task_id,
    }
    return await get_task_papers_handler(params, None)


GET_TASK_PAPERS_TOOL = get_task_papers_tool
