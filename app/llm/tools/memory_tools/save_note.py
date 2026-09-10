from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.llm.tools.memory_tools.common import fact_data, require_database
from app.llm.tools.registry import Tool, ToolExecutionContext
from app.memory.semantic.service import FactService
from app.rag.index_construction.memory_index_sync import (
    get_memory_index_synchronizer,
)


class SaveNoteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=1000)

    @field_validator("subject", "content")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


async def save_note_handler(
    params: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    args = SaveNoteArgs.model_validate(params)
    fact, created = FactService(
        require_database(context),
        user_id=context.user_id,
    ).remember_explicit(
        subject=args.subject,
        content=args.content,
        source_conversation_id=context.conversation_id or None,
    )
    if created:
        await get_memory_index_synchronizer().upsert_facts([fact])
    return {
        "ok": True,
        "data": {**fact_data(fact), "created": created},
        "message": "笔记已保存" if created else "相同笔记已存在",
    }


SAVE_NOTE_TOOL = Tool(
    name="save_note",
    description="将用户明确要求记住的信息保存为长期笔记。仅在用户明确要求保存、记住或记录时使用。",
    input_schema=SaveNoteArgs.model_json_schema(),
    fn=save_note_handler,
    requires_confirmation=True,
)
