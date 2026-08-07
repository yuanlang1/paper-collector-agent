from app.llm.subagents.contracts import SubAgentSpec
from app.llm.subagents.task_review.contracts import TaskReviewDelegation


def build_task_review_agent_spec() -> SubAgentSpec:
    return SubAgentSpec(
        name="task_review_agent",
        description=(
            "基于已完成 RAG 的检索任务论文集生成学术综述，不扩展论文集。"
        ),
        input_model=TaskReviewDelegation,
        requires_confirmation=True,
    )
