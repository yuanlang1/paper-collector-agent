from app.llm.subagents.contracts import SubAgentSpec
from app.llm.subagents.paper_search.contracts import (
    PaperSearchDelegation,
)


def build_paper_search_agent_spec() -> SubAgentSpec:
    return SubAgentSpec(
        name="paper_search_agent",
        description="检索、补充检索、筛选、推荐并保存学术论文。",
        input_model=PaperSearchDelegation,
        requires_confirmation=True,
    )
