from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import NotFoundException
from app.db.models import LlmSetting
from app.db.repositories import llm_setting


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    base_url: str
    model: str
    api_key: str


async def create_setting(
    db: AsyncSession,
    *,
    user_id: int,
    provider: str,
    base_url: str,
    model: str,
    api_key: str,
    is_small_model: bool = False,
) -> LlmSetting:
    async with db.begin():
        return await llm_setting.create(
            db,
            user_id=user_id,
            provider=provider,
            base_url=base_url,
            model=model,
            api_key=api_key,
            is_small_model=is_small_model,
        )


async def get_setting(
    db: AsyncSession,
    model_id: int,
    *,
    user_id: int,
) -> LlmSetting | None:
    async with db.begin():
        return await llm_setting.get_by_id(db, model_id, user_id=user_id)


async def list_settings(
    db: AsyncSession,
    *,
    user_id: int,
    is_small_model: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[LlmSetting]:
    async with db.begin():
        return await llm_setting.list_settings(
            db,
            user_id=user_id,
            is_small_model=is_small_model,
            limit=limit,
            offset=offset,
        )


async def update_setting(
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
    async with db.begin():
        return await llm_setting.update(
            db,
            model_id,
            user_id=user_id,
            provider=provider,
            base_url=base_url,
            model=model,
            api_key=api_key,
            is_small_model=is_small_model,
        )


async def delete_setting(db: AsyncSession, model_id: int, *, user_id: int) -> bool:
    async with db.begin():
        return await llm_setting.delete(db, model_id, user_id=user_id)


async def get_llm_config(db: AsyncSession, model_id: int, *, user_id: int) -> LlmConfig:
    setting = await get_setting(db, model_id, user_id=user_id)
    if setting is None:
        raise NotFoundException("LLM 配置不存在")
    return LlmConfig(
        provider=setting.provider,
        base_url=setting.base_url,
        model=setting.model,
        api_key=setting.api_key,
    )
