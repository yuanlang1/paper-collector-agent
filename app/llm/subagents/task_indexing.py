from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphBubbleUp, GraphInterrupt
from pydantic import BaseModel, Field

from app.bus.events import SubagentFinishedEvent, SubagentStartedEvent
from app.config import settings
from app.llm.subagents.paper_search import create_child_run_id


def _now():
    return datetime.now(UTC).isoformat()


class TaskIndexingParams(BaseModel):
    task_id: int = Field(
        gt = 0, 
        description = "需要构建知识库索引的已完成检索任务 ID。",
    )


@tool(
    "task_indexing_agent",
    description = (
        "将已完成论文检索任务中的论文构建索引并保存到知识库。"
        "当用户需要将任务论文入库，或为后续内容检索和综述生成准备 RAG 索引时，调用此工具。"
        "输入应为已完成检索任务的正整数 task_id。"
    ),
    args_schema = TaskIndexingParams,
)
async def task_indexing_agent(
    task_id: int,
    runtime: ToolRuntime,
) -> ToolMessage:
    """将已完成论文检索任务的论文索引并保存到知识库。"""
    args = {
        "task_id": task_id
    }
    parent_run_id = runtime.context.run_id
    session_id = runtime.config["configurable"]["thread_id"]
    run_id = create_child_run_id(parent_run_id)
    fields = {
        "session_id": session_id,
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "parent_session_id": session_id,
        "name": "task_indexing_agent",
    }
    writer = runtime.stream_writer
    writer(
        SubagentStartedEvent(
        **fields, ts = _now(), description = "开始执行知识库索引子代理。"
    ))
    try:
        from app.llm.graph.workflows.task_indexing.workflow import build_task_indexing_workflow

        output = await asyncio.wait_for(
            build_task_indexing_workflow().ainvoke(
                {
                    "run_id": parent_run_id,
                    "child_run_id": run_id,
                    "conversation_id": session_id,
                    **args,
                },
                config = runtime.config,
                context = {
                    "event_fields": fields
                },
            ),
            timeout = max(
                settings.SUBAGENT_DISPATCH_TIMEOUT_SECONDS, 
                settings.TASK_RAG_WORKFLOW_TIMEOUT_SECONDS + settings.SUBAGENT_DISPATCH_TIMEOUT_GRACE_SECONDS
            ),
        )
        if output.get("__interrupt__"):
            raise GraphInterrupt(tuple(output["__interrupt__"]))

        content = output["content"]
        status = output["status"]

    except GraphBubbleUp:
        raise

    except asyncio.CancelledError:
        writer(SubagentFinishedEvent(
            **fields,
            ts = _now(),
            description = "子代理执行已取消。",
            status = "cancelled",
        ))
        raise

    except Exception as exc:
        writer(SubagentFinishedEvent(
            **fields,
            ts = _now(),
            description = str(exc) or type(exc).__name__,
            status = "timed_out" if isinstance(exc, TimeoutError) else "failed",
        ))
        raise

    writer(SubagentFinishedEvent(
        **fields,
        ts = _now(),
        description = content,
        status = status,
    ))

    return ToolMessage(
        tool_call_id = runtime.tool_call_id,
        name = "task_indexing_agent",
        content = content,
        status = "success" if status == "success" else "error",
    )


task_indexing_agent.metadata = {
    "display_name": "知识库索引子代理"
}
