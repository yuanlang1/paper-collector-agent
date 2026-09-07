import asyncio
from time import monotonic
from typing import Any

from sqlalchemy.orm import Session

from app.infrastructure.grpc.paper_service_grpc_client import (
    paper_service_grpc_client,
)
from app.infrastructure.grpc.task_service_grpc_client import TaskState
from app.llm.tools.registry import Tool
from app.llm.tools.task_tools.search_task.args import StartTaskRagArgs
from app.llm.tools.task_tools.search_task.rag_status import (
    get_task_rag_status_handler,
)
from app.rag.processing.task_rag_batch_runner import (
    get_task_rag_batch_runner,
)


_STARTABLE_TASK_STATES = {
    TaskState.SEARCH_COMPLETED.name,
    TaskState.SEARCH_PARTIAL_COMPLETED.name,
}
_FAILED_TASK_STATES = {
    TaskState.SEARCH_FAILED.name,
    TaskState.CANCELLED.name,
    TaskState.RAG_FAILED.name,
}


def _status_value(status: dict[str, Any]) -> str:
    return str(status["result"]["task_status"] or "").upper()


def _failure(
    *,
    task_id: int,
    error: str,
    message: str,
    status: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": False,
        "result": {
            "taskId": task_id,
            "rag_status": status.get("result"),
        },
        "error": error,
        "message": message,
        "metadata": status.get("metadata"),
    }


async def _wait_for_rag_completion(
    *,
    task_id: int,
    poll_interval_seconds: int,
    timeout_seconds: int,
    db: Session,
    started: bool,
) -> dict[str, Any]:
    deadline = monotonic() + timeout_seconds

    while True:
        status = await get_task_rag_status_handler({"task_id": task_id}, db)
        if not status.get("ok"):
            return status

        result = status["result"]
        task_status = _status_value(status)
        failed_papers = result["summary"]["failed"]
        if task_status in _FAILED_TASK_STATES or failed_papers:
            return _failure(
                task_id=task_id,
                error="RAG_INDEX_FAILED",
                message="RAG 索引失败，已结束等待。",
                status=status,
            )

        if result["is_rag_complete"]:
            return {
                "ok": True,
                "result": {
                    "taskId": task_id,
                    "accepted": started,
                    "already_running_in_this_process": not started,
                    "rag_status": result,
                },
                "error": None,
                "metadata": status.get("metadata"),
            }

        remaining_seconds = deadline - monotonic()
        if remaining_seconds <= 0:
            return _failure(
                task_id=task_id,
                error="RAG_INDEX_TIMEOUT",
                message="等待 RAG 索引完成超时，已结束等待。",
                status=status,
            )

        await asyncio.sleep(min(poll_interval_seconds, remaining_seconds))


async def start_task_rag_handler(
    params: dict[str, Any],
    db: Session,
) -> dict[str, Any]:
    request = StartTaskRagArgs.model_validate(params)
    status = await get_task_rag_status_handler(
        {"task_id": request.taskId},
        db,
    )

    if not status.get("ok"):
        return status

    task_status = _status_value(status)
    if status["result"]["summary"]["failed"] or task_status in _FAILED_TASK_STATES:
        return _failure(
            task_id=request.taskId,
            error="RAG_INDEX_FAILED",
            message="RAG 索引已失败，已结束等待。",
            status=status,
        )

    if status["result"]["is_rag_complete"]:
        return {
            "ok": True,
            "result": {
                "taskId": request.taskId,
                "accepted": False,
                "already_running_in_this_process": False,
                "rag_status": status["result"],
            },
            "error": None,
            "metadata": status.get("metadata"),
        }

    if task_status == TaskState.RAG_RUNNING.name:
        started = False
    elif task_status in _STARTABLE_TASK_STATES:
        started = await get_task_rag_batch_runner().notify(request.taskId)
    else:
        return _failure(
            task_id=request.taskId,
            error="TASK_NOT_READY_FOR_RAG",
            message="任务尚未完成论文检索，无法启动 RAG 索引。",
            status=status,
        )

    return await _wait_for_rag_completion(
        task_id=request.taskId,
        poll_interval_seconds=request.poll_interval_seconds,
        timeout_seconds=request.timeout_seconds,
        db=db,
        started=started,
    )


START_TASK_RAG_TOOL = Tool(
    name="start_task_rag",
    description=(
        "启动指定论文检索任务的后台 RAG 流程，或接管已运行的 RAG 流程。"
        "工具会等待索引完成；索引失败、存在失败论文或超时会立即返回错误。"
    ),
    input_schema=StartTaskRagArgs.model_json_schema(),
    fn=start_task_rag_handler,
    requires_confirmation=True,
)
