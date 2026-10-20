from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import LlmSetting


async def create(
    db: AsyncSession,
    *,
    user_id: int,
    provider: str,
    base_url: str,
    model: str,
    api_key: str,
    is_small_model: bool = False,
) -> LlmSetting:
    setting = LlmSetting(
        user_id=user_id,
        provider=provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
        is_small_model=is_small_model,
    )
    db.add(setting)
    await db.flush()
    return setting


async def get_by_id(
    db: AsyncSession,
    model_id: int,
    *,
    user_id: int,
) -> LlmSetting | None:
    return await db.scalar(
        select(LlmSetting).where(
            LlmSetting.id == model_id,
            LlmSetting.user_id == user_id,
        )
    )


async def list_settings(
    db: AsyncSession,
    *,
    user_id: int,
    is_small_model: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[LlmSetting]:
    statement = select(LlmSetting).where(LlmSetting.user_id == user_id)
    if is_small_model is not None:
        statement = statement.where(LlmSetting.is_small_model == is_small_model)
    return list(
        await db.scalars(
            statement.order_by(LlmSetting.id.desc()).limit(limit).offset(offset)
        )
    )


async def update(
    db: AsyncSession,
    model_id: int,
    *,
    user_id: int,
    provider: str,
    base_url: str,
    model: str,
    api_key: str,
    is_small_model: bool,
) -> LlmSetting:
    setting = await get_by_id(db, model_id, user_id=user_id)
    if setting is None:
        raise ValueError("LLM setting not found")
    setting.provider = provider
    setting.base_url = base_url
    setting.model = model
    setting.api_key = api_key
    setting.is_small_model = is_small_model
    await db.flush()
    return setting


async def delete(db: AsyncSession, model_id: int, *, user_id: int) -> bool:
    setting = await get_by_id(db, model_id, user_id=user_id)
    if setting is None:
        return False
    await db.delete(setting)
    await db.flush()
    return True
