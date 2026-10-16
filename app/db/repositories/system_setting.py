from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SystemSetting


class SystemSettingRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get(self, key: str) -> SystemSetting | None:
        return await self.db.scalar(select(SystemSetting).where(SystemSetting.key == key))

    async def set_value(self, key: str, value: str) -> None:
        setting = await self.get(key)
        if setting is None:
            self.db.add(SystemSetting(key=key, value=value))
        else:
            setting.value = value
            setting.modifier_id = 0
        await self.db.flush()
