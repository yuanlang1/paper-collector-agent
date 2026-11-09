from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import ConflictException, NotFoundException
from app.database import SessionLocal
from app.db.repositories import llm_setting
from app.db.repositories.llm_setting import LlmSettingRow


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    base_url: str
    model: str
    api_key: str


def _raise_role_conflict(exc: IntegrityError) -> None:
    constraint = getattr(getattr(exc.orig, "diag", None), "constraint_name", None)
    if constraint in {
        "uq_llm_setting_user_default",
        "uq_llm_setting_user_small_model",
    }:
        raise ConflictException("模型标记发生并发冲突，请重试") from exc


async def create_setting(
    db: AsyncSession,
    *,
    user_id: int,
    provider: str,
    base_url: str,
    model: str,
    api_key: str,
    is_small_model: bool = False,
    is_default: bool = False,
) -> LlmSettingRow:
    try:
        async with db.begin():
            await llm_setting.lock_user_settings(db, user_id=user_id)
            await llm_setting.clear_role_flags(
                db,
                user_id=user_id,
                is_default=is_default,
                is_small_model=is_small_model,
            )
            return await llm_setting.create(
                db,
                user_id=user_id,
                provider=provider,
                base_url=base_url,
                model=model,
                api_key=api_key,
                is_small_model=is_small_model,
                is_default=is_default,
            )
    except IntegrityError as exc:
        _raise_role_conflict(exc)
        raise


async def get_setting(
    db: AsyncSession,
    model_id: int,
    *,
    user_id: int,
) -> LlmSettingRow | None:
    async with db.begin():
        return await llm_setting.get_by_id(db, model_id, user_id=user_id)


async def list_settings(
    db: AsyncSession,
    *,
    user_id: int,
    is_small_model: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[LlmSettingRow]:
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
    is_small_model: bool | None = None,
    is_default: bool | None = None,
) -> LlmSettingRow:
    try:
        async with db.begin():
            await llm_setting.lock_user_settings(db, user_id=user_id)

            if await llm_setting.get_by_id(db, model_id, user_id=user_id) is None:
                raise NotFoundException("LLM 配置不存在")

            await llm_setting.clear_role_flags(
                db,
                user_id=user_id,
                is_default=is_default is True,
                is_small_model=is_small_model is True,
                exclude_model_id=model_id,
            )

            return await llm_setting.update(
                db,
                model_id,
                user_id=user_id,
                provider=provider,
                base_url=base_url,
                model=model,
                api_key=api_key,
                is_small_model=is_small_model,
                is_default=is_default,
            )
    except IntegrityError as exc:
        _raise_role_conflict(exc)
        raise


async def delete_setting(db: AsyncSession, model_id: int, *, user_id: int) -> bool:
    async with db.begin():
        await llm_setting.lock_user_settings(db, user_id=user_id)
        return await llm_setting.delete(db, model_id, user_id=user_id)


@asynccontextmanager
async def _config_session(db: AsyncSession | None) -> AsyncIterator[AsyncSession]:
    if db is None:
        async with SessionLocal.begin() as session:
            yield session
    elif db.in_transaction():
        yield db
    else:
        async with db.begin():
            yield db


async def get_llm_config(
    model_id: int,
    *,
    user_id: int,
    db: AsyncSession | None = None,
) -> LlmConfig:
    async with _config_session(db) as session:
        setting = await llm_setting.get_by_id(session, model_id, user_id=user_id)
    if setting is None:
        raise NotFoundException("LLM 配置不存在")
    return LlmConfig(
        provider=setting.provider,
        base_url=setting.base_url,
        model=setting.model,
        api_key=setting.api_key,
    )


async def get_default_model(
    *,
    user_id: int,
    db: AsyncSession | None = None,
) -> LlmConfig | None:
    async with _config_session(db) as session:
        setting = await llm_setting.get_default_model(session, user_id=user_id)
        if setting is None:
            return None

        return LlmConfig(
            provider=setting.provider,
            base_url=setting.base_url,
            model=setting.model,
            api_key=setting.api_key,
        )


async def get_small_llm_config(
    *,
    user_id: int,
    db: AsyncSession | None = None,
) -> LlmConfig:
    async with _config_session(db) as session:
        setting = await llm_setting.get_small_model(session, user_id=user_id)
        if setting is None:
            settings = await llm_setting.list_settings(
                session, user_id=user_id, limit=1
            )
            if not settings:
                raise NotFoundException("小模型配置不存在")
            setting = settings[0]

        return LlmConfig(
            provider=setting.provider,
            base_url=setting.base_url,
            model=setting.model,
            api_key=setting.api_key,
        )
