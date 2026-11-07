from sqlalchemy.ext.asyncio import AsyncSession

from app.services.setting_service import (
    get_source_limits,
    save_source_limits,
    source_limit_maxima,
    source_page_sizes,
)


async def get_limits(db: AsyncSession) -> dict[str, dict[str, int]]:
    return {
        "limits": await get_source_limits(db),
        "page_sizes": source_page_sizes(),
        "maximum_limits": source_limit_maxima(),
    }


async def update_limits(db: AsyncSession, limits: dict[str, int]) -> dict[str, dict[str, int]]:
    return {
        "limits": await save_source_limits(db, limits),
        "page_sizes": source_page_sizes(),
        "maximum_limits": source_limit_maxima(),
    }
