from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel


@dataclass(frozen=True)
class SubAgentSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    requires_confirmation: bool = True
