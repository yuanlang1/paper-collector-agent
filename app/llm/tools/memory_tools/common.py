from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.llm.tools.registry import ToolExecutionContext
from app.models.memory import MemoryFact


def require_database(context: ToolExecutionContext) -> Session:
    if context.db is None:
        raise RuntimeError("Tool execution requires a database session")
    return context.db


def fact_data(fact: MemoryFact) -> dict[str, Any]:
    return {
        "id": fact.id,
        "subject": fact.subject,
        "content": fact.content,
        "status": fact.status,
        "updated_at": fact.updated_at.isoformat(),
    }
