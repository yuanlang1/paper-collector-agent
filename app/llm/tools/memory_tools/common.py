from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.tools.registry import ToolExecutionContext


def require_database(context: ToolExecutionContext) -> AsyncSession:
    if context.db is None:
        raise RuntimeError("Tool execution requires a database session")
    return context.db
