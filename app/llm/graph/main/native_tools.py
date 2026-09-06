from typing import Any, Literal

from app.llm.subagents.registry import ALL_SUBAGENTS, SUBAGENT_BY_NAME
from app.llm.tools.registry import ToolRegistry, build_tool_registry


ToolKind = Literal["tool", "subagent"]


def build_native_tool_schemas(
    tool_registry: ToolRegistry | None = None,
) -> list[dict[str, Any]]:
    registry = tool_registry or build_tool_registry()
    return [
        *registry.schemas(),
        *(subagent.to_api() for subagent in ALL_SUBAGENTS),
    ]


def get_tool_kind(
    name: str,
    tool_registry: ToolRegistry | None = None,
) -> ToolKind | None:
    registry = tool_registry or build_tool_registry()
    if registry.get(name) is not None:
        return "tool"
    if name in SUBAGENT_BY_NAME:
        return "subagent"
    return None


def requires_confirmation(
    name: str,
    tool_registry: ToolRegistry | None = None,
) -> bool:
    registry = tool_registry or build_tool_registry()
    tool = registry.get(name)
    if tool is not None:
        return tool.requires_confirmation
    subagent = SUBAGENT_BY_NAME.get(name)
    return subagent.requires_confirmation if subagent else False
