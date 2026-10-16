from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SystemSetting


async def get(db: AsyncSession, key: str) -> SystemSetting | None:
    return await db.scalar(select(SystemSetting).where(SystemSetting.key == key))


async def set_value(
    db: AsyncSession,
    key: str,
    value: str,
) -> None:
    setting = await get(db, key)
    if setting is None:
        db.add(SystemSetting(key=key, value=value))
    else:
        setting.value = value
        setting.modifier_id = 0
    await db.flush()
