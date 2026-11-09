from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import ConflictException, NotFoundException
from app.core.identifiers import normalize_user_id
from app.db.models.search_connection import SearchConnectionStatus
from app.db.repositories import search_connection as repository
from app.schemas.search_connection import (
    CreateSearchConnection,
    SearchConnectionResponse,
    UpdateSearchConnection,
)
from app.utils.search_adapters.contracts import DEFAULT_INITIAL_LIMITS


def _user_id(value: int | str) -> int:
    return normalize_user_id(value)[1]


def _response(row) -> SearchConnectionResponse:
    if row is None:
        raise NotFoundException("检索连接不存在")
    return SearchConnectionResponse.model_validate(row)


async def create_connection(
    db: AsyncSession, *, user_id: int | str, data: CreateSearchConnection
) -> SearchConnectionResponse:
    data = CreateSearchConnection.model_validate(data)
    values = data.model_dump()
    if not values["settings"] and data.provider_id in DEFAULT_INITIAL_LIMITS:
        values["settings"] = {"initial_limit": DEFAULT_INITIAL_LIMITS[data.provider_id]}
    try:
        async with db.begin():
            return _response(
                await repository.create(
                    db, user_id=_user_id(user_id), values=values
                )
            )
    except IntegrityError as exc:
        constraint = getattr(getattr(exc.orig, "diag", None), "constraint_name", None)
        if constraint == "uq_search_connection_user_provider":
            raise ConflictException("该检索源已配置，请修改已有连接") from exc
        raise


async def get_connection(
    db: AsyncSession, *, user_id: int | str, connection_id: int
) -> SearchConnectionResponse:
    async with db.begin():
        return _response(
            await repository.get_by_id(
                db, user_id=_user_id(user_id), connection_id=connection_id
            )
        )


async def get_connection_by_provider(
    db: AsyncSession, *, user_id: int | str, provider_id: str
) -> SearchConnectionResponse:
    async with db.begin():
        return _response(
            await repository.get_by_provider(
                db, user_id=_user_id(user_id), provider_id=provider_id
            )
        )


async def list_connections(
    db: AsyncSession,
    *,
    user_id: int | str,
    provider_id: str | None = None,
    enabled: bool | None = None,
    status: SearchConnectionStatus | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[SearchConnectionResponse]:
    if not 1 <= limit <= 100 or offset < 0:
        raise ValueError("limit must be 1-100 and offset must be non-negative")
    if status is not None:
        status = SearchConnectionStatus(status)
    async with db.begin():
        rows = await repository.list_connections(
            db,
            user_id=_user_id(user_id),
            provider_id=provider_id,
            enabled=enabled,
            status=status,
            limit=limit,
            offset=offset,
        )
        return [_response(row) for row in rows]


async def update_connection(
    db: AsyncSession,
    *,
    user_id: int | str,
    connection_id: int,
    data: UpdateSearchConnection,
) -> SearchConnectionResponse:
    data = UpdateSearchConnection.model_validate(data)
    values = data.model_dump(exclude_unset=True)
    user_id = _user_id(user_id)
    async with db.begin():
        if values.get("settings") == {}:
            current = await repository.get_by_id(db, user_id=user_id, connection_id=connection_id)
            if current is None:
                raise NotFoundException("检索连接不存在")
            if current.provider_id in DEFAULT_INITIAL_LIMITS:
                values["settings"] = {"initial_limit": DEFAULT_INITIAL_LIMITS[current.provider_id]}
        return _response(
            await repository.update_connection(
                db,
                user_id=user_id,
                connection_id=connection_id,
                values=values,
            )
        )


async def delete_connection(
    db: AsyncSession, *, user_id: int | str, connection_id: int
) -> None:
    async with db.begin():
        if not await repository.delete_connection(
            db, user_id=_user_id(user_id), connection_id=connection_id
        ):
            raise NotFoundException("检索连接不存在")


async def set_connection_status(
    db: AsyncSession,
    *,
    user_id: int | str,
    connection_id: int,
    status: SearchConnectionStatus,
) -> SearchConnectionResponse:
    status = SearchConnectionStatus(status)
    async with db.begin():
        return _response(
            await repository.update_connection(
                db,
                user_id=_user_id(user_id),
                connection_id=connection_id,
                values={"status": status},
            )
        )
