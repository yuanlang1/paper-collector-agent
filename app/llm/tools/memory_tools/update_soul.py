from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, tool
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


class UpdateSoulToolArgs(UpdateSoulArgs):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    runtime: ToolRuntime[Any] = Field(exclude=True)


async def update_soul_handler(
    params: dict[str, Any],
    runtime: ToolRuntime,
    *,
    session_factory: async_sessionmaker[AsyncSession],
) -> dict[str, Any]:
    args = UpdateSoulArgs.model_validate(params)
    async with session_factory() as db:
        preference, created = await UserPreferenceService(
            db,
            user_id=runtime.context.user_id,
        ).add(
            instruction=args.instruction,
            source_conversation_id=runtime.context.session_id or None,
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


def build_update_soul_tool(
    *,
    session_factory: async_sessionmaker[AsyncSession],
) -> BaseTool:
    @tool(
        "update_soul",
        description=(
            "保存当前用户的长期交互偏好。"
            "仅当用户明确要求长期记住回答风格、语言或其他交互偏好时，调用此工具。"
            "输入 instruction 应为简洁明确的偏好说明，不用于修改共享的 Agent 固定准则。"
        ),
        args_schema=UpdateSoulToolArgs,
    )
    async def update_soul_tool(
        *,
        runtime: ToolRuntime,
        instruction: str,
    ) -> dict[str, Any]:
        """保存当前用户明确提出的长期交互偏好；不会修改共享的 Agent 固定准则。"""
        params = {
            "instruction": instruction,
        }
        return await update_soul_handler(params, runtime, session_factory=session_factory)

    return update_soul_tool
