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
from app.llm.tools.registry import Tool, ToolExecutionContext


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

    return await task_service_grpc_client.add_query_task(payload)


ADD_QUERY_TASK_TOOL = Tool(
    name="add_query_task",
    description=(
        "创建并保存论文查询任务。"
        "该工具会通过 Nacos 发现 paper-service，"
        "调用 addQueryTask 接口持久化用户的原始检索要求、"
        "检索标签和结构化查询理解。"
        "仅当用户明确要求创建、提交、启动或保存论文查询任务时使用；"
        "如果用户只是咨询检索方案、要求推荐关键词或预览查询条件，"
        "不得调用该工具。"
    ),
    input_schema=AddQueryTaskArgs.model_json_schema(),
    fn=add_query_task_handler,
    requires_confirmation=True,
)
