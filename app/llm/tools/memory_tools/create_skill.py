from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import settings
from app.llm.tools.registry import Tool, ToolExecutionContext
from app.memory.procedural.loader import (
    Skill,
    render_skill_document,
    user_skill_directory,
)


SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62})$")


class CreateSkillArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=63)
    description: str = Field(min_length=1, max_length=400)
    triggers: list[str] = Field(min_length=1, max_length=8)
    instructions: str = Field(min_length=1, max_length=5000)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip().lower()
        if not SKILL_NAME_PATTERN.fullmatch(value):
            raise ValueError("name must use lowercase letters, digits or hyphens")
        return value

    @field_validator("description", "instructions")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("description")
    @classmethod
    def require_single_line_description(cls, value: str) -> str:
        if "\n" in value or "\r" in value:
            raise ValueError("description must be single-line")
        return value

    @field_validator("triggers")
    @classmethod
    def normalize_triggers(cls, values: list[str]) -> list[str]:
        normalized = []
        for value in values:
            value = value.strip()
            if not value or "\n" in value or "," in value:
                raise ValueError("each trigger must be non-empty and single-line")
            normalized.append(value)
        return list(dict.fromkeys(normalized))


async def create_skill_handler(
    params: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    args = CreateSkillArgs.model_validate(params)
    skill = Skill(
        name=args.name,
        description=args.description,
        triggers=tuple(args.triggers),
        body=args.instructions,
    )
    directory = user_skill_directory(
        Path(settings.AGENT_SKILLS_DIR),
        context.user_id,
    )
    path = directory / skill.name / "SKILL.md"
    if path.exists():
        raise FileExistsError("同名 Skill 已存在，未覆盖原文件")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_skill_document(skill), encoding="utf-8")
    return {
        "ok": True,
        "data": {
            "name": skill.name,
            "triggers": list(skill.triggers),
            "path": str(path),
        },
        "message": "Skill 已创建，将在下一轮匹配触发词时生效",
    }


CREATE_SKILL_TOOL = Tool(
    name="create_skill",
    description="为当前用户创建可复用的本地工作流 Skill。仅在用户明确要求沉淀或创建可复用流程时使用。",
    input_schema=CreateSkillArgs.model_json_schema(),
    fn=create_skill_handler,
    requires_confirmation=True,
)
