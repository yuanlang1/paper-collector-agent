from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import BadRequestException
from app.common.response import ServiceResponse
from app.database import get_db
from app.db.models.search_connection import SearchConnectionStatus
from app.schemas.search_connection import (
    CreateSearchConnection,
    ProviderId,
    SearchConnectionResponse,
    UpdateSearchConnection,
)
from app.schemas.setting import ModelSettingListItem, PostAddModelRequest
from app.services import llm_setting_service, search_connection_service
from app.utils.search_adapters.registry import supported_provider_ids

router = APIRouter(prefix="/api/setting", tags=["setting"])


@router.post("/model_setting")
async def add_model(
    body: PostAddModelRequest, db: Annotated[AsyncSession, Depends(get_db)]
):
    llm_config = await llm_setting_service.create_setting(
        db,
        user_id=body.user_id,
        provider=body.provider,
        base_url=body.base_url,
        model=body.model_name,
        api_key=body.api_key,
        is_small_model=body.is_small_model,
        is_default=body.is_default,
    )

    return ServiceResponse.build_success_response(data=llm_config.id)


UserId = Annotated[int, Query(gt=0, le=9_223_372_036_854_775_807)]
ConnectionId = Annotated[int, Path(gt=0, le=9_223_372_036_854_775_807)]


@router.get("/model_setting")
async def list_models(
    user_id: UserId,
    db: Annotated[AsyncSession, Depends(get_db)],
    is_small_model: bool | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ServiceResponse[list[ModelSettingListItem]]:
    models = await llm_setting_service.list_settings(
        db, user_id=user_id, is_small_model=is_small_model, limit=limit, offset=offset
    )
    return ServiceResponse[list[ModelSettingListItem]].build_success_response(
        [ModelSettingListItem.model_validate(model) for model in models]
    )


@router.post("/search-connections")
async def create_search_connection(
    user_id: UserId,
    body: CreateSearchConnection,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ServiceResponse[SearchConnectionResponse]:
    if body.provider_id not in supported_provider_ids():
        raise BadRequestException("不支持的检索源")
    connection = await search_connection_service.create_connection(
        db, user_id=user_id, data=body,
    )
    return ServiceResponse[SearchConnectionResponse].build_success_response(connection)


@router.get("/search-connections")
async def list_search_connections(
    user_id: UserId,
    db: Annotated[AsyncSession, Depends(get_db)],
    provider_id: Annotated[ProviderId | None, Query()] = None,
    enabled: bool | None = None,
    status: SearchConnectionStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ServiceResponse[list[SearchConnectionResponse]]:
    connections = await search_connection_service.list_connections(
        db, user_id=user_id, provider_id=provider_id, enabled=enabled,
        status=status, limit=limit, offset=offset,
    )
    return ServiceResponse[list[SearchConnectionResponse]].build_success_response(connections)


@router.get("/search-connections/{connection_id}")
async def get_search_connection(
    user_id: UserId,
    connection_id: ConnectionId,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ServiceResponse[SearchConnectionResponse]:
    connection = await search_connection_service.get_connection(
        db, user_id=user_id, connection_id=connection_id,
    )
    return ServiceResponse[SearchConnectionResponse].build_success_response(connection)


@router.patch("/search-connections/{connection_id}")
async def update_search_connection(
    user_id: UserId,
    connection_id: ConnectionId,
    body: UpdateSearchConnection,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ServiceResponse[SearchConnectionResponse]:
    connection = await search_connection_service.update_connection(
        db, user_id=user_id, connection_id=connection_id, data=body,
    )
    return ServiceResponse[SearchConnectionResponse].build_success_response(connection)


@router.delete("/search-connections/{connection_id}")
async def delete_search_connection(
    user_id: UserId,
    connection_id: ConnectionId,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    await search_connection_service.delete_connection(
        db, user_id=user_id, connection_id=connection_id,
    )
    return ServiceResponse.build_success_response(
        data={"connection_id": connection_id, "deleted": True},
    )
