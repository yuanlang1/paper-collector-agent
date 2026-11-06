from typing import Any

from sqlalchemy import case, delete, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.search_connection import SearchConnection, SearchConnectionStatus


async def create(db: AsyncSession, *, user_id: int, values: dict[str, Any]) -> SearchConnection:
    row = SearchConnection(user_id=user_id, **values)
    db.add(row)
    await db.flush()
    return row


async def get_by_id(db: AsyncSession, *, user_id: int, connection_id: int) -> SearchConnection | None:
    return await db.scalar(select(SearchConnection).where(
        SearchConnection.id == connection_id, SearchConnection.user_id == user_id,
    ))


async def get_by_provider(db: AsyncSession, *, user_id: int, provider_id: str) -> SearchConnection | None:
    return await db.scalar(select(SearchConnection).where(
        SearchConnection.user_id == user_id, SearchConnection.provider_id == provider_id,
    ))


async def list_connections(
    db: AsyncSession, *, user_id: int, provider_id: str | None = None,
    enabled: bool | None = None, status: str | None = None,
    limit: int = 50, offset: int = 0,
) -> list[SearchConnection]:
    statement = select(SearchConnection).where(SearchConnection.user_id == user_id)
    for name, value in (("provider_id", provider_id), ("enabled", enabled), ("status", status)):
        if value is not None:
            statement = statement.where(getattr(SearchConnection, name) == value)
    return list(await db.scalars(statement.order_by(
        SearchConnection.creation_time, SearchConnection.id,
    ).limit(limit).offset(offset)))


async def update_connection(
    db: AsyncSession, *, user_id: int, connection_id: int, values: dict[str, Any],
) -> SearchConnection | None:
    if not values:
        return await get_by_id(db, user_id=user_id, connection_id=connection_id)
    changes = [getattr(SearchConnection, name).is_distinct_from(values[name])
               for name in ("settings", "api_key") if name in values]
    if changes:
        values = {**values, "status": case(
            (or_(*changes), SearchConnectionStatus.PENDING), else_=SearchConnection.status,
        )}
    return await db.scalar(update(SearchConnection).where(
        SearchConnection.id == connection_id, SearchConnection.user_id == user_id,
    ).values(**values).returning(SearchConnection).execution_options(populate_existing=True))


async def delete_connection(db: AsyncSession, *, user_id: int, connection_id: int) -> bool:
    result = await db.scalar(delete(SearchConnection).where(
        SearchConnection.id == connection_id, SearchConnection.user_id == user_id,
    ).returning(SearchConnection.id))
    return result is not None
