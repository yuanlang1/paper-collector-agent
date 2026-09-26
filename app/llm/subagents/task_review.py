from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field
from app.llm.provider import ChatClient

if TYPE_CHECKING:
    from app.llm.subagents.registry import SubAgentRuntime


class TaskReviewDelegation(BaseModel):
    task_id: int = Field(gt=0)
    topic: str = Field(min_length=2, max_length=2_000)
    language: Literal["zh-CN", "en"] = "zh-CN"
    citation_style: Literal[
        "harvard",
        "apa",
        "ieee",
        "chicago",
        "vancouver",
    ] = "harvard"
    review_type: Literal[
        "narrative",
        "systematic",
        "scoping",
        "critical",
    ] = "narrative"
    max_reflection_rounds: int = Field(default=5, ge=1, le=5)
    max_writing_revision_rounds: int = Field(default=1, ge=0, le=5)

def build_task_review_runtime(*, chat: ChatClient) -> SubAgentRuntime:
    from app.llm.graph.workflows.review_generate.workflow import (
        build_task_review_workflow,
    )
    from app.llm.subagents.registry import (
        SubAgentRuntime,
        SubAgentSpec,
        SubAgentStreamSpec,
    )
    return SubAgentRuntime(
        spec=SubAgentSpec(
            name="task_review_agent",
            description=(
                "基于已完成 RAG 的固定任务论文集生成结构化学术综合，不扩展论文集。"
                "systematic/scoping 仅指定组织方式，不代表执行了正式系统综述检索与筛选协议。"
            ),
            input_model=TaskReviewDelegation,
            requires_confirmation=True,
            display_name="文献综述子代理",
            confirmation_summary="将分析已有文献并生成综述建议。",
        ),
        graph=build_task_review_workflow(chat=chat),
        error_code="TASK_REVIEW_SUBGRAPH_FAILED",
        failure_summary="文献综述工作流执行失败，未完成结果保存。",
        stream=SubAgentStreamSpec(workflow="task_review"),
    )
