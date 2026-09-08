from typing import Any

from sqlalchemy.orm import Session

from app.llm.tools.registry import Tool
from app.llm.tools.task_tools.search_task.args import GetTaskRagStatusArgs
from app.rag.processing.task_rag_status import get_task_rag_status


async def get_task_rag_status_handler(
    params: dict[str, Any],
    _db: Session,
) -> dict[str, Any]:
    request = GetTaskRagStatusArgs.model_validate(params)
    return await get_task_rag_status(request.task_id)


GET_TASK_RAG_STATUS_TOOL = Tool(
    name = "get_task_rag_status",
    description = (
        "查询论文检索任务的RAG索引状态，返回任务状态、各论文 RAG 状态统计，以及当前是否可以生成文献综述。"
    ),
    input_schema = GetTaskRagStatusArgs.model_json_schema(),
    fn = get_task_rag_status_handler,
)
