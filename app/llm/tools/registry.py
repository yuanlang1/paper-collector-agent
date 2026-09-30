
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.llm.artifacts.access import ArtifactAccessService
from app.events.context import current_event_context
from app.events.errors import EventPublicationError
from app.llm.tools.base import BaseTool, ToolResult


@dataclass(frozen=True)
class ToolExecutionContext:
    db: Session | None
    user_id: str = "0"
    conversation_id: str = ""
    run_id: str = ""
    artifact_access: ArtifactAccessService | None = None


class ToolRegistry:
    def __init__(self, db: Session | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._db = db

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.to_api() for tool in self._tools.values()]

    async def execute(
        self,
        name: str,
        args: dict[str, Any],
        *,
        context: ToolExecutionContext | None = None,
        timeout: float = 20,
    ) -> ToolResult:
        tool_context = context or ToolExecutionContext(db=self._db)
        event_context = current_event_context()
        await event_context.bus.publish(
            event_context.event(
                "tool_started",
                {"message": "工具开始执行。", "progress": 0, "data": {}},
            ),
        )
        try:
            tool = self.get(name)
            if tool is None:
                raise LookupError(f"未知工具：{name}")
            if tool.params_model is not None:
                try:
                    tool.params_model.model_validate(dict(args))
                except ValidationError as exc:
                    return await self._fail(
                        event_context,
                        error_message=str(exc),
                        error_name=type(exc).__name__,
                    )

            result = await asyncio.wait_for(tool.fn(args, tool_context), timeout)

            await event_context.bus.publish(
                event_context.event(
                    "tool_completed",
                    {
                        "message": result.content,
                        "progress": 100,
                        "data": {},
                    },
                ),
            )
        except asyncio.TimeoutError:
            return await self._fail(
                event_context,
                error_message=f"工具执行超时（{timeout:g}s）。",
                error_name="timeout",
            )
        except EventPublicationError:
            raise
        except Exception as exc:
            return await self._fail(
                event_context,
                error_message=str(exc),
                error_name=type(exc).__name__,
            )

        return result

    async def _fail(
        self,
        event_context,
        *,
        error_message: str,
        error_name: str,
    ) -> ToolResult:
        result = ToolResult(
            content=error_message,
            is_error=True,
            error_type=error_name,
        )
        await event_context.bus.publish(
            event_context.event(
                "tool_failed",
                {"message": result.content, "progress": None, "data": {}},
            ),
        )
        return result


def build_tool_registry(db: Session | None = None) -> ToolRegistry:
    from app.llm.tools.artifact_tools.read_artifact import READ_ARTIFACT_TOOL
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
        READ_ARTIFACT_TOOL,
        SAVE_NOTE_TOOL,
        MANAGE_MEMORY_TOOL,
        UPDATE_SOUL_TOOL,
        CREATE_SKILL_TOOL,
    ):
        registry.register(tool)
    return registry
