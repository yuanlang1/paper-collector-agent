import json
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories import system_setting


async def get_setting(
    db: AsyncSession,
    key: str,
    default: Any = None,
) -> Any:
    async with db.begin():
        value = await system_setting.get(db, key)
    if value:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return default


async def save_setting(
    db: AsyncSession,
    key: str,
    value: Any,
) -> None:
    serialized = json.dumps(value, ensure_ascii=False)
    async with db.begin():
        await system_setting.set_value(
            db,
            key,
            serialized,
        )


async def get_custom_system_prompt(db: AsyncSession) -> str:
    return (
        await get_setting(
            db,
            "agent_system_prompt",
            "",
        )
    ) or ""
