from app.llm.subagents.contracts import SubAgentSpec
from app.llm.subagents.paper_search.agent import (
    build_paper_search_agent_spec,
)
from app.llm.subagents.task_review.agent import (
    build_task_review_agent_spec,
)


ALL_SUBAGENTS: tuple[SubAgentSpec, ...] = (
    build_paper_search_agent_spec(),
    build_task_review_agent_spec(),
)

SUBAGENT_BY_NAME: dict[str, SubAgentSpec] = {
    spec.name: spec for spec in ALL_SUBAGENTS
}


class SubAgentRegistry:
    def __init__(self, specs: list[SubAgentSpec] | tuple[SubAgentSpec, ...]):
        self._specs = {spec.name: spec for spec in specs}

    def all(self) -> tuple[SubAgentSpec, ...]:
        return tuple(self._specs.values())

    def get(self, name: str | None) -> SubAgentSpec | None:
        return self._specs.get(name) if name else None

    def descriptions(self) -> str:
        return "\n".join(
            f"- {spec.name}: {spec.description}"
            for spec in self._specs.values()
        )
