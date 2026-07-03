from app.llm.tools.base import AppToolSpec

ALL_TOOLS: list[AppToolSpec] = [
]

TOOL_BY_NAME: dict[str, AppToolSpec] = {
    tool.name: tool for tool in ALL_TOOLS
}

def get_tool_specs(
        allowed_tools: list[str] | None = None
    ) -> list[AppToolSpec]:
    if allowed_tools is None:
        return ALL_TOOLS

    allowed = set(allowed_tools)
    return [tool for tool in ALL_TOOLS if tool.name in allowed]

def tool_descriptions() -> str:
    return "\n".join(
        f"- {tool.name}: {tool.description}"
        for tool in ALL_TOOLS
    )

def requires_confirmation(
        tool_names: list[str]
    ) -> bool:
    return any(
        TOOL_BY_NAME[name].requires_confirmation
        for name in tool_names
        if name in TOOL_BY_NAME
    )