from app.llm.graph.workflows.review_generate.workflow import (
    build_task_review_workflow,
)
from app.llm.subagents.registry import (
    SubAgentRuntime,
    SubAgentSpec,
    SubAgentStreamSpec,
)
from app.llm.subagents.task_review.contracts import TaskReviewDelegation


def build_task_review_runtime() -> SubAgentRuntime:
    return SubAgentRuntime(
        spec=SubAgentSpec(
            name="task_review_agent",
            description=(
                "基于已完成 RAG 的检索任务论文集生成学术综述，不扩展论文集。"
            ),
            input_model=TaskReviewDelegation,
            requires_confirmation=True,
            display_name="文献综述子代理",
            confirmation_summary="将分析已有文献并生成综述建议。",
        ),
        graph=build_task_review_workflow(),
        error_code="TASK_REVIEW_SUBGRAPH_FAILED",
        failure_summary="文献综述工作流执行失败，未完成结果保存。",
        stream=SubAgentStreamSpec(
            workflow="task_review",
            phases=(
                ("prepare", "初始化与语料校验"),
                ("framework", "生成综述框架"),
                ("claims", "生成论点"),
                ("evidence", "检索证据"),
                ("render", "撰写章节"),
                ("review", "反思与修订"),
                ("finalize", "生成最终综述"),
            ),
            node_phases={
                "initialize": "prepare",
                "load_task_corpus": "prepare",
                "generate_framework": "framework",
                "generate_claims": "claims",
                "retrieve_evidence": "evidence",
                "render_sections": "render",
                "assemble_review": "render",
                "reflect_review": "review",
                "finalizing_handoff": "finalize",
                "finalize_result": "finalize",
            },
            iteration_key="reflection_round",
        ),
    )
