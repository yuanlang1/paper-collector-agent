from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from app.config import settings
from app.llm.provider import ChatClient
from app.llm.tools.base import BaseTool

if TYPE_CHECKING:
    from app.llm.subagents.registry import SubAgentRuntime

class PaperSearchParams(BaseModel):
    prompt: str = Field(min_length = 1, max_length = 2_000)
    objective: str = "检索、筛选、推荐并保存相关论文"
    task_id: int | None = Field(default = None, gt = 0, strict = True)

class SearchPapersTool(BaseTool):
    name = "paper_search_agent"
    description = (
        "检索、补充检索、筛选、推荐并保存学术论文。"
        "若要复用已有检索任务，请通过 task_id 传入其正整数 ID。"
    )
    params_model = PaperSearchParams

def build_paper_search_runtime(
    *,
    chat: ChatClient,
) -> SubAgentRuntime:
    from app.llm.graph.workflows.paper_search.workflow import (
        build_paper_search_workflow,
    )
    from app.llm.subagents.registry import (
        SubAgentRuntime,
        SubAgentSpec,
        SubAgentStreamSpec,
    )
    return SubAgentRuntime(
        spec=SubAgentSpec(
            name="paper_search_agent",
            description=(
                "检索、补充检索、筛选、推荐并保存学术论文。"
                "若要复用已有检索任务，请通过 task_id 传入其正整数 ID。"
            ),
            input_model=PaperSearchDelegation,
            requires_confirmation=False,
            display_name="论文检索子代理",
        ),
        graph=build_paper_search_workflow(chat=chat),
        error_code="PAPER_SEARCH_SUBGRAPH_FAILED",
        failure_summary="论文检索工作流执行失败，未完成结果保存。",
        stream=SubAgentStreamSpec(workflow="paper_search"),
        timeout_seconds=settings.SUBAGENT_DISPATCH_TIMEOUT_SECONDS,
    )
