from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class LlmSettingRow:
    id: int
    user_id: int
    provider: str
    base_url: str
    model: str
    api_key: str
    is_small_model: bool
    is_default: bool
    creator_id: int
    creation_time: datetime
    modifier_id: int
    modification_time: datetime


async def lock_user_settings(db: AsyncSession, *, user_id: int) -> None:
    await db.execute(
        text("""
            SELECT pg_advisory_xact_lock(CAST(:user_id AS BIGINT))
        """),
        {"user_id": user_id},
    )


async def create(
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
    result = await db.execute(
        text("""
            INSERT INTO llm_setting
                (user_id, provider, base_url, model, api_key, is_small_model, is_default)
            VALUES
                (:user_id, :provider, :base_url, :model, :api_key, :is_small_model, :is_default)
            RETURNING *
        """),
        {
            "user_id": user_id,
            "provider": provider,
            "base_url": base_url,
            "model": model,
            "api_key": api_key,
            "is_small_model": is_small_model,
            "is_default": is_default,
        },
    )
    return LlmSettingRow(**result.mappings().one())


async def get_by_id(
    db: AsyncSession,
    model_id: int,
    *,
    user_id: int,
) -> LlmSettingRow | None:
    result = await db.execute(
        text("""
            SELECT *
            FROM llm_setting
            WHERE id = :model_id
              AND user_id = :user_id
        """),
        {"model_id": model_id, "user_id": user_id},
    )
    row = result.mappings().first()
    return LlmSettingRow(**row) if row is not None else None


async def get_small_model(db: AsyncSession, *, user_id: int) -> LlmSettingRow | None:
    result = await db.execute(
        text("""
            SELECT *
            FROM llm_setting
            WHERE user_id = :user_id
              AND is_small_model = TRUE
            ORDER BY id DESC
            LIMIT 1
        """),
        {"user_id": user_id},
    )
    row = result.mappings().first()
    return LlmSettingRow(**row) if row is not None else None


async def get_default_model(db: AsyncSession, *, user_id: int) -> LlmSettingRow | None:
    result = await db.execute(
        text("""
            SELECT *
            FROM llm_setting
            WHERE user_id = :user_id
              and is_default = TRUE
            ORDER BY id DESC
            LIMIT 1
        """),
        {"user_id": user_id},
    )
    row = result.mappings().first()
    return LlmSettingRow(**row) if row is not None else None


async def list_settings(
    db: AsyncSession,
    *,
    user_id: int,
    is_small_model: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[LlmSettingRow]:
    small_filter = (
        "" if is_small_model is None else "AND is_small_model = :is_small_model"
    )
    result = await db.execute(
        text(f"""
            SELECT *
            FROM llm_setting
            WHERE user_id = :user_id
              {small_filter}
            ORDER BY id DESC
            LIMIT :limit
            OFFSET :offset
        """),
        {
            "user_id": user_id,
            "is_small_model": is_small_model,
            "limit": limit,
            "offset": offset,
        },
    )
    return [LlmSettingRow(**row) for row in result.mappings()]


async def clear_role_flags(
    db: AsyncSession,
    *,
    user_id: int,
    is_default: bool,
    is_small_model: bool,
    exclude_model_id: int | None = None,
) -> None:
    if not is_default and not is_small_model:
        return
    await db.execute(
        text("""
            UPDATE llm_setting
            SET is_default = CASE WHEN :is_default THEN FALSE ELSE is_default END,
                is_small_model = CASE WHEN :is_small_model THEN FALSE ELSE is_small_model END,
                modification_time = now()
            WHERE user_id = :user_id
              AND id IS DISTINCT FROM CAST(:exclude_model_id AS BIGINT)
              AND ((:is_default AND is_default IS TRUE)
                   OR (:is_small_model AND is_small_model IS TRUE))
        """),
        {
            "user_id": user_id,
            "is_default": is_default,
            "is_small_model": is_small_model,
            "exclude_model_id": exclude_model_id,
        },
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
    is_small_model: bool | None = None,
    is_default: bool | None = None,
) -> LlmSettingRow:
    result = await db.execute(
        text("""
            UPDATE llm_setting
            SET provider = :provider,
                base_url = :base_url,
                model = :model,
                api_key = :api_key,
                is_small_model = COALESCE(CAST(:is_small_model AS BOOLEAN), is_small_model),
                is_default = COALESCE(CAST(:is_default AS BOOLEAN), is_default),
                modification_time = CASE
                    WHEN (
                        provider,
                        base_url,
                        model,
                        api_key,
                        is_small_model,
                        is_default
                    ) IS DISTINCT FROM (
                        CAST(:provider AS VARCHAR),
                        CAST(:base_url AS VARCHAR),
                        CAST(:model AS VARCHAR),
                        CAST(:api_key AS TEXT),
                        COALESCE(CAST(:is_small_model AS BOOLEAN), is_small_model),
                        COALESCE(CAST(:is_default AS BOOLEAN), is_default)
                    )
                    THEN now()
                    ELSE modification_time
                END
            WHERE id = :model_id
              AND user_id = :user_id
            RETURNING *
        """),
        {
            "model_id": model_id,
            "user_id": user_id,
            "provider": provider,
            "base_url": base_url,
            "model": model,
            "api_key": api_key,
            "is_small_model": is_small_model,
            "is_default": is_default,
        },
    )
    setting = result.mappings().first()
    if setting is None:
        raise ValueError("LLM setting not found")
    return LlmSettingRow(**setting)


async def delete(db: AsyncSession, model_id: int, *, user_id: int) -> bool:
    result = await db.execute(
        text("""
            DELETE FROM llm_setting
            WHERE id = :model_id
              AND user_id = :user_id
            RETURNING id
        """),
        {"model_id": model_id, "user_id": user_id},
    )
    return result.scalar_one_or_none() is not None
