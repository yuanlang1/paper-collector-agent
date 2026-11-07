from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def get(db: AsyncSession, key: str) -> str | None:
    result = await db.execute(
        text("""
            SELECT value
            FROM config_setting
            WHERE key = :key
        """),
        {"key": key},
    )
    return result.scalar_one_or_none()


async def set_value(
    db: AsyncSession,
    key: str,
    value: str,
) -> None:
    await db.execute(
        text("""
            INSERT INTO config_setting (key, value)
            VALUES (:key, :value)
            ON CONFLICT (key) DO UPDATE
            SET value = EXCLUDED.value,
                modifier_id = 0,
                modification_time = now()
            WHERE config_setting.value IS DISTINCT FROM EXCLUDED.value
               OR config_setting.modifier_id IS DISTINCT FROM 0
        """),
        {"key": key, "value": value},
    )
