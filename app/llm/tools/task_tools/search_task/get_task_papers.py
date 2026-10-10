from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from app.infrastructure.grpc.paper_service_grpc_client import (
    paper_service_grpc_client,
)
from app.llm.tools.base import BaseTool, ToolResult
from app.llm.tools.registry import ToolExecutionContext


class GetTaskPapersArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: StrictInt = Field(
        ..., gt=0, description="要查看论文的检索任务 ID。"
    )


async def get_task_papers_handler(
    params: dict[str, Any],
    context: ToolExecutionContext,
) -> ToolResult:
    del context
    request = GetTaskPapersArgs.model_validate(params)
    result = await paper_service_grpc_client.get_task_review_papers(
        request.task_id
    )
    if not result.get("ok"):
        raise RuntimeError(str(result.get("error") or "任务论文查询失败。"))

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
    return ToolResult(
        content=json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


GET_TASK_PAPERS_TOOL = BaseTool(
    name="get_task_papers",
    description=(
        "获取指定检索任务下的全部论文，返回标题、作者、摘要和 PDF 地址。"
        "用于查看这个任务主要有哪些论文、了解论文涉及的研究主题，以及依据任务中的论文推荐综述题目或研究方向。"
        "不要求 RAG 索引完成。"
    ),
    input_schema=GetTaskPapersArgs.model_json_schema(),
    fn=get_task_papers_handler,
    params_model=GetTaskPapersArgs,
)
