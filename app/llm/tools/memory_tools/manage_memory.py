from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.llm.tools.memory_tools.common import fact_data, require_database
from app.llm.tools.registry import Tool, ToolExecutionContext
from app.memory.semantic.service import FactService


class ManageMemoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["search", "update", "delete"]
    query: str | None = Field(default=None, min_length=1, max_length=300)
    fact_id: int | None = Field(default=None, ge=1)
    subject: str | None = Field(default=None, min_length=1, max_length=200)
    content: str | None = Field(default=None, min_length=1, max_length=1000)

    @field_validator("query", "subject", "content")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_action_arguments(self) -> "ManageMemoryArgs":
        if self.action == "search" and self.query is None:
            raise ValueError("query is required for search")
        if self.action == "update" and (
            self.fact_id is None
            or self.subject is None
            or self.content is None
        ):
            raise ValueError("fact_id, subject and content are required for update")
        if self.action == "delete" and self.fact_id is None:
            raise ValueError("fact_id is required for delete")
        return self


async def manage_memory_handler(
    params: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    args = ManageMemoryArgs.model_validate(params)
    facts = FactService(require_database(context), user_id=context.user_id)

    if args.action == "search":
        return {
            "ok": True,
            "data": {
                "facts": [
                    fact_data(fact)
                    for fact in facts.search_active(query=args.query or "")
                ]
            },
            "message": "已查询长期记忆",
        }

    if args.action == "update":
        fact = facts.correct(
            fact_id=args.fact_id or 0,
            subject=args.subject or "",
            content=args.content or "",
        )
        return {
            "ok": True,
            "data": fact_data(fact),
            "message": "长期记忆已更新",
        }

    fact = facts.forget(fact_id=args.fact_id or 0)
    return {
        "ok": True,
        "data": fact_data(fact),
        "message": "长期记忆已删除",
    }


MANAGE_MEMORY_TOOL = Tool(
    name="manage_memory",
    description="查询、更新或删除当前用户的长期笔记。更新和删除前必须获得用户确认。",
    input_schema=ManageMemoryArgs.model_json_schema(),
    fn=manage_memory_handler,
    requires_confirmation=lambda args: args.get("action") in {"update", "delete"},
)
