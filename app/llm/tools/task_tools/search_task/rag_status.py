from collections import Counter
from typing import Any

from sqlalchemy.orm import Session

from app.infrastructure.grpc.paper_service_grpc_client import (
    paper_service_grpc_client,
)
from app.llm.tools.registry import Tool
from app.llm.tools.task_tools.search_task.args import GetTaskRagStatusArgs


async def get_task_rag_status_handler(
    params: dict[str, Any],
    _db: Session,
) -> dict[str, Any]:
    request = GetTaskRagStatusArgs.model_validate(params)
    response = await paper_service_grpc_client.get_task_review_papers(
        request.task_id,
    )

    if not response["ok"]:
        return response

    result = response["result"]
    counts = Counter(paper["rag_status"] for paper in result["papers"])
    is_rag_complete = (
        result["task_status"].lower() == "rag_completed"
        and not counts["pending"]
        and not counts["indexing"]
        and not counts["failed"]
    )

    return {
        "ok": True,
        "result": {
            "task_id": result["task_id"],
            "task_status": result["task_status"],
            "summary": {
                "total": len(result["papers"]),
                "pending": counts["pending"],
                "indexing": counts["indexing"],
                "ready": counts["ready"],
                "skipped": counts["skipped"],
                "failed": counts["failed"],
                "ready_chunks": sum(
                    paper["chunk_count"]
                    for paper in result["papers"]
                    if paper["rag_status"] == "ready"
                ),
            },
            "is_rag_complete": is_rag_complete,
            "can_generate_review": is_rag_complete and bool(counts["ready"]),
        },
        "error": None,
        "metadata": response["metadata"],
    }


GET_TASK_RAG_STATUS_TOOL = Tool(
    name = "get_task_rag_status",
    description = (
        "查询论文检索任务的RAG索引状态，返回任务状态、各论文 RAG 状态统计，以及当前是否可以生成文献综述。"
    ),
    input_schema = GetTaskRagStatusArgs.model_json_schema(),
    fn = get_task_rag_status_handler,
)
