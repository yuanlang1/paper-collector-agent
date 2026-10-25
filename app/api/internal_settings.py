from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import BadRequestException
from app.common.response import ServiceResponse
from app.database import get_db
from app.schemas.setting import UpdateSourceLimitsRequest
from app.services import internal_settings_service

router = APIRouter(prefix="/internal/settings", tags=["internal-settings"])


@router.get("/source-limits")
async def get_source_limits(db: Annotated[AsyncSession, Depends(get_db)]):
    return ServiceResponse.build_success_response(
        await internal_settings_service.get_limits(db)
    )


@router.put("/source-limits")
async def update_source_limits(
    body: UpdateSourceLimitsRequest, db: Annotated[AsyncSession, Depends(get_db)],
):
    try:
        values = await internal_settings_service.update_limits(db, body.model_dump())
    except ValueError as exc:
        raise BadRequestException(str(exc)) from exc
    return ServiceResponse.build_success_response(values)
