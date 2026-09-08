from typing import Any, Literal

from app.llm.subagents.registry import SubAgentRegistry
from app.llm.tools.registry import ToolRegistry


ToolKind = Literal["tool", "subagent"]


def build_native_tool_schemas(
    tool_registry: ToolRegistry,
    subagent_registry: SubAgentRegistry,
) -> list[dict[str, Any]]:
    return [
        *tool_registry.schemas(),
        *(subagent.to_api() for subagent in subagent_registry.all_specs()),
    ]


def get_tool_kind(
    name: str,
    tool_registry: ToolRegistry,
    subagent_registry: SubAgentRegistry,
) -> ToolKind | None:
    if tool_registry.get(name) is not None:
        return "tool"
    if subagent_registry.get_spec(name) is not None:
        return "subagent"
    return None


def requires_confirmation(
    name: str,
    args: dict[str, Any],
    tool_registry: ToolRegistry,
    subagent_registry: SubAgentRegistry,
) -> bool:
    tool = tool_registry.get(name)
    if tool is not None:
        policy = tool.requires_confirmation
        return policy(args) if callable(policy) else policy
    subagent = subagent_registry.get_spec(name)
    return subagent.requires_confirmation if subagent else False
