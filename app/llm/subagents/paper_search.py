from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.errors import GraphBubbleUp, GraphInterrupt
from langgraph.prebuilt.tool_node import ToolRuntime
from pydantic import BaseModel, Field, model_validator

from app.bus.events import SubagentFinishedEvent, SubagentStartedEvent
from app.config import settings
from app.core.identifiers import normalize_user_id
from app.infrastructure.paper_service.http_client import positive_id


def create_child_run_id(parent_run_id: str) -> str:
    parent_digest = hashlib.sha256(parent_run_id.encode("utf-8")).hexdigest()[:16]

    return f"child_{parent_digest}_{uuid4().hex}"


def _now():
    return datetime.now(UTC).isoformat()


class PaperSearchParams(BaseModel):
    prompt: str = Field(
        min_length=1,
        max_length=2_000,
        description="用户原始论文检索要求，包含研究主题及用户明确提出的限制。",
    )

    provider_ids: list[str] | None = Field(
        default=None,
        min_length=1,
        description="可选检索来源 ID；不指定时使用当前用户已启用的可用来源。",
    )

    @model_validator(mode="after")
    def validate_providers(self):
        if self.provider_ids is not None:
            if len(set(self.provider_ids)) != len(self.provider_ids):
                raise ValueError("provider_ids must not contain duplicates")
            from app.utils.search_adapters.registry import get_adapter

            for provider in self.provider_ids:
                get_adapter(provider)
        return self


def build_paper_search_tool(*, model: BaseChatModel) -> BaseTool:
    @tool(
        "paper_search_agent",
        description=(
            "检索、补充检索、筛选、推荐并保存学术论文。"
            "当用户需要查找研究主题的相关论文、了解研究进展或获取论文推荐时，调用此工具。"
            "输入 prompt 应保留用户原始检索要求，包含明确的研究主题或关键词及用户提出的限制。"
        ),
        args_schema=PaperSearchParams,
    )
    async def paper_search_agent(
        prompt: str,
        runtime: ToolRuntime,
        provider_ids: list[str] | None = None,
    ) -> ToolMessage:
        args = {"prompt": prompt, "provider_ids": provider_ids}
        parent_run_id = runtime.context.run_id
        session_id = runtime.config["configurable"]["thread_id"]
        run_id = create_child_run_id(parent_run_id)

        fields = {
            "session_id": session_id,
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "parent_session_id": session_id,
            "name": "paper_search_agent",
        }

        writer = runtime.stream_writer
        writer(
            SubagentStartedEvent(
                **fields, ts=_now(), description="开始执行论文检索子代理。"
            )
        )

        try:
            user_id, numeric_user_id = normalize_user_id(runtime.context.user_id)
            positive_id(numeric_user_id, "user_id")
            from app.llm.graph.workflows.paper_search.workflow import (
                build_paper_search_workflow,
            )

            output = await asyncio.wait_for(
                build_paper_search_workflow(model=model).ainvoke(
                    {
                        "run_id": parent_run_id,
                        "child_run_id": run_id,
                        "conversation_id": session_id,
                        **args,
                    },
                    config={
                        **runtime.config,
                        "recursion_limit": 10007,
                        "max_concurrency": min(
                            runtime.config.get("max_concurrency") or 3, 3
                        ),
                    },
                    context={"event_fields": fields, "user_id": user_id},
                ),
                timeout=settings.SUBAGENT_DISPATCH_TIMEOUT_SECONDS,
            )
            if output.get("__interrupt__"):
                raise GraphInterrupt(tuple(output["__interrupt__"]))

            content = output["content"]
            status = output["status"]

        except GraphBubbleUp:
            raise

        except asyncio.CancelledError:
            writer(
                SubagentFinishedEvent(
                    **fields,
                    ts=_now(),
                    description="子代理执行已取消。",
                    status="cancelled",
                )
            )
            raise

        except Exception as exc:
            writer(
                SubagentFinishedEvent(
                    **fields,
                    ts=_now(),
                    description=str(exc) or type(exc).__name__,
                    status="timed_out" if isinstance(exc, TimeoutError) else "failed",
                )
            )
            raise

        writer(
            SubagentFinishedEvent(
                **fields,
                ts=_now(),
                description=content,
                status=status,
            )
        )

        return ToolMessage(
            tool_call_id=runtime.tool_call_id,
            name="paper_search_agent",
            content=content,
            status="success" if status == "success" else "error",
        )

    paper_search_agent.metadata = {
        "display_name": "论文检索子代理",
    }

    return paper_search_agent
