from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.llm.subagents.registry import SubAgentRuntime


class TaskIndexingDelegation(BaseModel):
    task_id: int = Field(gt=0)
    timeout_seconds: int = Field(default=900, ge=1, le=3_600)
    poll_interval_seconds: int = Field(default=5, ge=1, le=60)


def build_task_indexing_runtime() -> SubAgentRuntime:
    from app.llm.graph.workflows.task_indexing.workflow import (
        build_task_indexing_workflow,
    )
    from app.llm.subagents.registry import (
        SubAgentRuntime,
        SubAgentSpec,
        SubAgentStreamSpec,
    )

    return SubAgentRuntime(
        spec=SubAgentSpec(
            name="task_indexing_agent",
            description="将已完成论文检索任务的论文索引并保存到知识库。",
            input_model=TaskIndexingDelegation,
            requires_confirmation=True,
            display_name="知识库索引子代理",
            confirmation_summary="将解析论文并写入知识库。",
        ),
        graph=build_task_indexing_workflow(),
        error_code="TASK_INDEXING_SUBGRAPH_FAILED",
        failure_summary="知识库索引工作流执行失败，未确认索引完成。",
        stream=SubAgentStreamSpec(
            workflow="task_indexing",
            phases=(
                ("prepare", "校验任务状态"),
                ("index", "索引论文到知识库"),
                ("verify", "确认索引结果"),
                ("finalize", "汇总索引结果"),
            ),
            node_phases={
                "initialize": "prepare",
                "check_rag_status": "prepare",
                "run_or_wait": "index",
                "verify_completion": "verify",
                "finalize_result": "finalize",
            },
            detail_state_keys={
                "overall_summary": "overall_summary",
                "committed_summary": "committed_summary",
                "overall_progress_percent": "overall_progress_percent",
                "committed_progress_percent": "committed_progress_percent",
                "error_code": "error_code",
            },
        ),
    )
