from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar
from langgraph.types import StreamWriter

from pydantic import BaseModel


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    error_type: str | None = None

@dataclass
class ToolContext:
    writer: StreamWriter | None = None
    user_id: str | None = None
    session_id: str | None = None

class BaseTool:
    name: str
    description: str
    input_schemas: dict[str, object]
    params_model: ClassVar[type[BaseModel] | None] = None

    @abstractmethod
    async def invoke(
        self, 
        params: dict[str, object],
        *,
        context: ToolContext | None = None,
    ) -> ToolResult:
        ...
