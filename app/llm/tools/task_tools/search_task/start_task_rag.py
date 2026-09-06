from typing import Any

from sqlalchemy.orm import Session

from app.infrastructure.grpc.paper_service_grpc_client import (
    paper_service_grpc_client,
)
from app.infrastructure.grpc.task_service_grpc_client import TaskState
from app.llm.tools.registry import Tool
from app.llm.tools.task_tools.search_task.args import StartTaskRagArgs
from app.rag.processing.task_rag_batch_runner import (
    get_task_rag_batch_runner,
)


async def start_task_rag_handler(
    params: dict[str, Any],
    _db: Session,
) -> dict[str, Any]:
    request = StartTaskRagArgs.model_validate(params)
    response = await paper_service_grpc_client.get_task_review_papers(
        request.taskId,
    )

    if not response["ok"]:
        return response

    task_status = response["result"]["task_status"]
    if task_status != TaskState.SEARCH_COMPLETED.name:
        return {
            "ok": False,
            "result": {
                "taskId": request.taskId,
                "task_status": task_status,
            },
            "error": "TASK_NOT_SEARCH_COMPLETED",
            "message": (
                "RAG 流程只能在任务状态为 SEARCH_COMPLETED 时启动。"
            ),
            "metadata": response["metadata"],
        }

    started = await get_task_rag_batch_runner().notify(request.taskId)
    return {
        "ok": True,
        "result": {
            "taskId": request.taskId,
            "accepted": started,
            "already_running_in_this_process": not started,
        },
        "error": None,
        "metadata": response["metadata"],
    }


START_TASK_RAG_TOOL = Tool(
    name="start_task_rag",
    description=(
        "启动指定论文检索任务的后台 RAG 流程。"
        "调用前任务状态必须为 SEARCH_COMPLETED；"
        "该操作会下载、解析论文并写入索引，启动后使用 get_task_rag_status 查询进度。"
    ),
    input_schema=StartTaskRagArgs.model_json_schema(),
    fn=start_task_rag_handler,
    requires_confirmation=True,
)
