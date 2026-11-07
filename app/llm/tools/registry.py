from __future__ import annotations

from langchain_core.tools import BaseTool
from sqlalchemy.ext.asyncio import AsyncSession


class ToolRegistry:
    def __init__(
        self, 
        db: AsyncSession | None = None
    ) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._tools_name: dict[str, str] = {}

    def register(
        self, 
        tool: BaseTool,
        name: str,
    ) -> None:
        self._tools[tool.name] = tool
        self._tools_name[tool.name] = name

    def get_tools(self) -> list[BaseTool]:
        return self._tools.values()

    def get_preview_name(
        self,
        name: str
    ) -> str:
        preview_name = self._tools_name.get(name)
        if preview_name == None:
            return name
        
        else:
            return preview_name
    