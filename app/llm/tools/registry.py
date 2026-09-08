
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session


ToolHandler = Callable[[dict[str, Any], Session | None], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: ToolHandler
    requires_confirmation: bool = False

    def to_api(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(self, db: Session | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._db = db

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.to_api() for tool in self._tools.values()]

    async def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool = self.get(name)
        if tool is None:
            return {"ok": False, "error": "UNKNOWN_TOOL"}

        params = {
            key: value
            for key, value in args.items()
            if value is not None and value != "" and value != []
        }
        try:
            result = await tool.fn(params, self._db)
        except Exception as exc:
            return {
                "ok": False,
                "error": type(exc).__name__,
                "message": str(exc),
            }

        return result if isinstance(result, dict) else {"result": result}


def build_tool_registry(db: Session | None = None) -> ToolRegistry:
    from app.llm.tools.file_tools.download_file import DOWNLOAD_FILE_TOOL
    from app.llm.tools.search_tools.arxiv.search_arxiv import ARXIV_SEARCH_TOOL
    from app.llm.tools.search_tools.crossref.search_crossref import CROSSREF_SEARCH_TOOL
    from app.llm.tools.search_tools.dblp.search_dblp import DBLP_SEARCH_TOOL
    from app.llm.tools.search_tools.google_scholar.search_google_scholar import (
        GOOGLE_SCHOLAR_SEARCH_TOOL,
    )
    from app.llm.tools.task_tools.search_task.rag_status import GET_TASK_RAG_STATUS_TOOL
    from app.llm.tools.task_tools.search_task.search_task_create import ADD_QUERY_TASK_TOOL
    from app.llm.tools.venue_tools.easy_scholar import EASY_SCHOLAR_VENUE_TOOL

    registry = ToolRegistry(db=db)
    for tool in (
        ARXIV_SEARCH_TOOL,
        DBLP_SEARCH_TOOL,
        CROSSREF_SEARCH_TOOL,
        GOOGLE_SCHOLAR_SEARCH_TOOL,
        ADD_QUERY_TASK_TOOL,
        GET_TASK_RAG_STATUS_TOOL,
        EASY_SCHOLAR_VENUE_TOOL,
        DOWNLOAD_FILE_TOOL,
    ):
        registry.register(tool)
    return registry
