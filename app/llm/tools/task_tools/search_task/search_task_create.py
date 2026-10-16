from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.infrastructure.grpc.task_service_grpc_client import (
    task_service_grpc_client,
)
from app.llm.graph.workflows.paper_search_schemas import (
    NonBlankString,
    PromptUnderstandingArgs,
    SearchTagArgs,
)
from langchain_core.tools import tool
from app.llm.tools.registry import ToolExecutionContext


class AddQueryTaskArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: NonBlankString = Field(..., description="用户原始的论文检索要求。")
    searchTag: SearchTagArgs
    queryUnderstanding: PromptUnderstandingArgs


async def add_query_task_handler(
    params: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    del context
    request = AddQueryTaskArgs.model_validate(params)

    payload = request.model_dump(
        mode="json",
        exclude_none=False,
    )

    result = await task_service_grpc_client.add_query_task(payload)
    if not result.get("ok"):
        raise RuntimeError(str(result.get("error") or "创建检索任务失败。"))
    return result


@tool(
    "add_query_task",
    description=(
        "通过 paper-service 创建并保存论文查询任务。"
        "仅当用户明确要求创建、提交、启动或保存论文查询任务时，调用此工具。"
        "输入应包含用户原始检索要求 prompt、检索标签 searchTag 和结构化查询理解 queryUnderstanding。"
        "用户仅咨询检索方法、推荐关键词或预览查询条件时，不调用此工具。"
    ),
    args_schema=AddQueryTaskArgs,
)
async def add_query_task_tool(
    *,
    prompt: NonBlankString,
    searchTag: SearchTagArgs,
    queryUnderstanding: PromptUnderstandingArgs,
) -> dict[str, Any]:
    """创建并保存论文查询任务。该工具会通过 Nacos 发现 paper-service，调用 addQueryTask 接口持久化用户的原始检索要求、检索标签和结构化查询理解。仅当用户明确要求创建、提交、启动或保存论文查询任务时使用；如果用户只是咨询检索方案、要求推荐关键词或预览查询条件，不得调用该工具。"""
    params = {
        "prompt": prompt,
        "searchTag": searchTag,
        "queryUnderstanding": queryUnderstanding,
    }
    return await add_query_task_handler(params, None)


ADD_QUERY_TASK_TOOL = add_query_task_tool
ADD_QUERY_TASK_TOOL.metadata = {"requires_confirmation": True}
