from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Literal

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphBubbleUp, GraphInterrupt
from pydantic import BaseModel, Field

from app.bus.events import SubagentFinishedEvent, SubagentStartedEvent
from app.config import settings
from app.llm.provider import ChatClient
from app.llm.subagents.paper_search import create_child_run_id

def _now():
    return datetime.now(UTC).isoformat()

class TaskReviewParams(BaseModel):
    task_id: int = Field(
        gt=0, description="已完成 RAG 索引、用于提供固定论文集的检索任务 ID。",
    )
    topic: str = Field(
        min_length=2, max_length=2_000, description="基于任务论文集撰写综述的主题。",
    )
    language: Literal["zh-CN", "en"] = Field(
        default="zh-CN", description="综述输出语言：zh-CN 为中文，en 为英文；默认中文。",
    )
    citation_style: Literal[
        "harvard",
        "apa",
        "ieee",
        "chicago",
        "vancouver",
    ] = Field(
        default="harvard",
        description="综述引用格式：harvard、apa、ieee、chicago 或 vancouver；默认 harvard。",
    )
    review_type: Literal[
        "narrative",
        "systematic",
        "scoping",
        "critical",
    ] = Field(
        default="narrative",
        description="综述组织方式：narrative、systematic、scoping 或 critical；仅指定组织方式，不代表执行正式系统综述检索协议。",
    )


@tool(
    "task_review_agent",
    description=(
        "基于已完成 RAG 索引的固定任务论文集生成结构化学术综述。"
        "当用户需要围绕任务论文撰写综述、综合研究进展或批判性分析时，调用此工具。"
        "输入应包含 task_id 和明确的综述主题 topic，可指定输出语言、引用格式和综述组织方式。"
        "不扩展论文集合；systematic 和 scoping 仅指定组织方式，不代表执行正式系统综述检索协议。"
    ),
    args_schema=TaskReviewParams,
)
async def task_review_agent(
    task_id: int,
    topic: str,
    runtime: ToolRuntime,
    language: Literal["zh-CN", "en"] = "zh-CN",
    citation_style: Literal["harvard", "apa", "ieee", "chicago", "vancouver"] = "harvard",
    review_type: Literal["narrative", "systematic", "scoping", "critical"] = "narrative",
) -> ToolMessage:
    args = {
        "task_id": task_id, "topic": topic, "language": language,
        "citation_style": citation_style, "review_type": review_type,
    }
    parent_run_id = runtime.context.run_id
    session_id = runtime.config["configurable"]["thread_id"]
    run_id = create_child_run_id(parent_run_id)
    fields = {
        "session_id": session_id, 
        "run_id": run_id,
        "parent_run_id": parent_run_id, 
        "parent_session_id": session_id,
        "name": "task_review_agent",
    }
    writer = runtime.stream_writer
    writer(
        SubagentStartedEvent(
            **fields, 
            ts = _now(), 
            description = "开始执行文献综述子代理。"
        )
    )
    try:
        from app.llm.graph.workflows.review_generate.workflow import build_task_review_workflow

        output = await asyncio.wait_for(
            build_task_review_workflow(chat = ChatClient()).ainvoke(
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
            timeout = settings.TASK_REVIEW_SUBAGENT_TIMEOUT_SECONDS,
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
        name = "task_review_agent",
        content = content,
        status = "success" if status == "success" else "error",
    )


task_review_agent.metadata = {
    "requires_confirmation": False, 
    "display_name": "文献综述子代理"
}
