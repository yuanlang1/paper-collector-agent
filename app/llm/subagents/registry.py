from app.llm.subagents.contracts import SubAgentSpec


class SubAgentRegistry:
    def __init__(self, specs: list[SubAgentSpec]):
        self._specs = {spec.name: spec for spec in specs}

    def get(self, name: str | None) -> SubAgentSpec | None:
        return self._specs.get(name) if name else None

    def descriptions(self) -> str:
        return "\n".join(
            f"- {spec.name}: {spec.description}"
            for spec in self._specs.values()
        )
