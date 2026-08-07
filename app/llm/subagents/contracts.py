from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.llm.tool_schema import build_model_tool_schema


@dataclass(frozen=True)
class SubAgentSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    requires_confirmation: bool = True

    def to_model_schema(self) -> dict[str, Any]:
        return build_model_tool_schema(
            name=self.name,
            description=self.description,
            input_model=self.input_model,
        )
