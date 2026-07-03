from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Type

from pydantic import BaseModel
from sqlalchemy.orm import Session


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
