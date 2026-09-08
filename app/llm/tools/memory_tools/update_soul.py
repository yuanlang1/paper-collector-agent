from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.llm.tools.memory_tools.common import require_database
from app.llm.tools.registry import Tool, ToolExecutionContext
from app.memory.preferences import UserPreferenceService


class UpdateSoulArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(min_length=1, max_length=300)

    @field_validator("instruction")
    @classmethod
    def strip_instruction(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("instruction must not be blank")
        return value


async def update_soul_handler(
    params: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    args = UpdateSoulArgs.model_validate(params)
    preference, created = UserPreferenceService(
        require_database(context),
        user_id=context.user_id,
    ).add(
        instruction=args.instruction,
        source_conversation_id=context.conversation_id or None,
    )
    return {
        "ok": True,
        "data": {
            "preference_id": preference.id,
            "instruction": preference.content,
            "created": created,
        },
        "message": "个人交互偏好已保存" if created else "相同个人交互偏好已存在",
    }


UPDATE_SOUL_TOOL = Tool(
    name="update_soul",
    description="保存当前用户明确提出的长期交互偏好；不会修改共享的 Agent 固定准则。",
    input_schema=UpdateSoulArgs.model_json_schema(),
    fn=update_soul_handler,
    requires_confirmation=True,
)
