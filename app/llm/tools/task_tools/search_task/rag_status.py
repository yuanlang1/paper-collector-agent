from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from langchain_core.tools import ToolException, tool
from app.rag.processing.task_rag_status import get_task_rag_status


class GetTaskRagStatusArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: int = Field(..., gt=0, description="要查询 RAG 状态的检索任务 ID。")


async def get_task_rag_status_handler(
    params: dict[str, Any],
    context: object | None = None,
) -> dict[str, Any]:
    del context
    request = GetTaskRagStatusArgs.model_validate(params)
    result = await get_task_rag_status(request.task_id)
    if not result.get("ok"):
        raise ToolException(str(result.get("error") or "RAG 状态查询失败。"))
    return result


@tool(
    "get_task_rag_status",
    description=(
        "查询论文检索任务及其论文的 RAG 索引状态和统计。"
        "当需要了解任务索引进度，或判断当前是否满足生成综述的条件时，调用此工具。"
        "输入应为需要查询的检索任务正整数 task_id。"
    ),
    args_schema=GetTaskRagStatusArgs,
)
async def get_task_rag_status_tool(
    *,
    task_id: int,
) -> dict[str, Any]:
    """查询论文检索任务的RAG索引状态，返回任务状态、各论文 RAG 状态统计，以及当前是否可以生成文献综述。"""
    params = {
        "task_id": task_id,
    }
    return await get_task_rag_status_handler(params, None)


GET_TASK_RAG_STATUS_TOOL = get_task_rag_status_tool
