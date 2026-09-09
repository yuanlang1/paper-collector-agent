
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.llm.streaming.notify import NOOP_NOTIFIER, Notifier


@dataclass(frozen=True)
class ToolExecutionContext:
    """Request-scoped resources and identity supplied to a tool call."""

    db: Session | None
    user_id: str = "0"
    conversation_id: str = ""
    run_id: str = ""
    _notify: Notifier = NOOP_NOTIFIER

    def notify(
        self,
        kind: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        self._notify(kind, payload)


ToolHandler = Callable[
    [dict[str, Any], ToolExecutionContext],
    Awaitable[dict[str, Any]],
]
ConfirmationPolicy = bool | Callable[[dict[str, Any]], bool]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: ToolHandler
    requires_confirmation: ConfirmationPolicy = False

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

    async def execute(
        self,
        name: str,
        args: dict[str, Any],
        *,
        context: ToolExecutionContext | None = None,
    ) -> dict[str, Any]:
        tool = self.get(name)
        if tool is None:
            return {"ok": False, "error": "UNKNOWN_TOOL"}

        params = {
            key: value
            for key, value in args.items()
            if value is not None and value != "" and value != []
        }
        try:
            result = await tool.fn(
                params,
                context or ToolExecutionContext(db=self._db),
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": type(exc).__name__,
                "message": str(exc),
            }

        return result if isinstance(result, dict) else {"result": result}


def build_tool_registry(db: Session | None = None) -> ToolRegistry:
    from app.llm.tools.file_tools.download_file import DOWNLOAD_FILE_TOOL
    from app.llm.tools.memory_tools.create_skill import CREATE_SKILL_TOOL
    from app.llm.tools.memory_tools.manage_memory import MANAGE_MEMORY_TOOL
    from app.llm.tools.memory_tools.save_note import SAVE_NOTE_TOOL
    from app.llm.tools.memory_tools.update_soul import UPDATE_SOUL_TOOL
    from app.llm.tools.search_tools.arxiv.search_arxiv import ARXIV_SEARCH_TOOL
    from app.llm.tools.search_tools.crossref.search_crossref import CROSSREF_SEARCH_TOOL
    from app.llm.tools.search_tools.dblp.search_dblp import DBLP_SEARCH_TOOL
    from app.llm.tools.search_tools.google_scholar.search_google_scholar import (
        GOOGLE_SCHOLAR_SEARCH_TOOL,
    )
    from app.llm.tools.web_tools.search_web import SEARCH_WEB_TOOL
    from app.llm.tools.task_tools.search_task.rag_status import GET_TASK_RAG_STATUS_TOOL
    from app.llm.tools.task_tools.search_task.search_task_create import ADD_QUERY_TASK_TOOL
    from app.llm.tools.venue_tools.easy_scholar import EASY_SCHOLAR_VENUE_TOOL

    registry = ToolRegistry(db=db)
    for tool in (
        ARXIV_SEARCH_TOOL,
        DBLP_SEARCH_TOOL,
        CROSSREF_SEARCH_TOOL,
        GOOGLE_SCHOLAR_SEARCH_TOOL,
        SEARCH_WEB_TOOL,
        ADD_QUERY_TASK_TOOL,
        GET_TASK_RAG_STATUS_TOOL,
        EASY_SCHOLAR_VENUE_TOOL,
        DOWNLOAD_FILE_TOOL,
        SAVE_NOTE_TOOL,
        MANAGE_MEMORY_TOOL,
        UPDATE_SOUL_TOOL,
        CREATE_SKILL_TOOL,
    ):
        registry.register(tool)
    return registry
