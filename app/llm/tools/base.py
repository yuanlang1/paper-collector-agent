from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Type

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.llm.tool_schema import build_model_tool_schema


ToolHandler = Callable[
    [Dict[str, Any], Session], 
    Awaitable[Dict[str, Any]]
]


@dataclass(frozen=True)
class AppToolSpec:
    name: str
    description: str
    args_schema: Type[BaseModel]
    handler: ToolHandler
    requires_confirmation: bool = False

    def to_model_schema(self) -> dict[str, Any]:
        return build_model_tool_schema(
            name=self.name,
            description=self.description,
            input_model=self.args_schema,
        )
