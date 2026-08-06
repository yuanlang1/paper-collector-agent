from typing import Any, Literal

from app.llm.subagents.registry import (
    ALL_SUBAGENTS,
    SUBAGENT_BY_NAME,
)
from app.llm.tools.registry import ALL_TOOLS, TOOL_BY_NAME

ToolKind = Literal["tool", "subagent"]

def build_native_tool_schemas() -> list[dict[str, Any]]:
    return [
        *(tool.to_model_schema() for tool in ALL_TOOLS),
        *(subagent.to_model_schema() for subagent in ALL_SUBAGENTS),
    ]


def get_tool_kind(name: str) -> ToolKind | None:
    if name in TOOL_BY_NAME:
        return "tool"
    if name in SUBAGENT_BY_NAME:
        return "subagent"
    return None


def requires_confirmation(name: str) -> bool:
    if name in TOOL_BY_NAME:
        return TOOL_BY_NAME[name].requires_confirmation
    subagent = SUBAGENT_BY_NAME.get(name)
    return subagent.requires_confirmation if subagent else False
