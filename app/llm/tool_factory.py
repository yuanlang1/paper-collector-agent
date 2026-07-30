from typing import Any

from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from app.llm.tools.registry import get_tool_specs


class ToolFactory:
    def build(
        self,
        db: Session,
        allowed_tools: list[str] | None = None,
    ) -> list[StructuredTool]:
        tools = []

        for spec in get_tool_specs(allowed_tools):
            async def _call(_spec=spec, **kwargs: Any):
                params = {
                    k: v for k, v in kwargs.items()
                    if v is not None and v != "" and v != []
                }
                return await _spec.handler(params, db)

            _call.__name__ = spec.name

            tools.append(
                StructuredTool.from_function(
                    coroutine=_call,
                    name=spec.name,
                    description=spec.description,
                    args_schema=spec.args_schema,
                )
            )

        return tools
