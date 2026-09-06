from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel



@dataclass(frozen=True)
class SubAgentSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    requires_confirmation: bool = True

    def to_api(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }
