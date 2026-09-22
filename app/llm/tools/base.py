from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False
    error_type: str | None = None


ToolService = Callable[[dict[str, Any], Any], Awaitable[dict[str, Any]]]
ToolHandler = Callable[[dict[str, Any], Any], Awaitable[ToolResult]]
ConfirmationPolicy = bool | Callable[[dict[str, Any]], bool]


def as_tool(service: ToolService) -> ToolHandler:
    async def handler(
        params: dict[str, Any],
        context: Any,
    ) -> ToolResult:
        return ToolResult(content=str(await service(params, context)))

    return handler


@dataclass(frozen=True)
class BaseTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: ToolHandler
    requires_confirmation: ConfirmationPolicy = False
    params_model: type[BaseModel] | None = None

    def to_api(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
