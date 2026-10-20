from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.response import ServiceResponse
from app.database import get_db
from app.schemas.setting import PostAddModelRequest
from app.services import llm_setting_service

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
    )

    return ServiceResponse.build_success_response(data=llm_config.id)
