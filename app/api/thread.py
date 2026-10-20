from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.response import ServiceResponse
from app.database import get_db
from app.schemas.ai import CreateThreadRequest
from app.services import thread_service

router = APIRouter(prefix="/api/thread", tags=["thread"])


@router.post("")
async def create_thread(
    body: CreateThreadRequest, db: Annotated[AsyncSession, Depends(get_db)]
):
    thread = await thread_service.create_thread(
        db, user_id=body.user_id, message=body.topic
    )
    return ServiceResponse.build_success_response(data=thread)


@router.get("/list")
async def get_thread_list(
    user_id: Annotated[int, Query(gt=0)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    threads = await thread_service.get_threads(
        db, user_id=user_id, limit=limit, offset=offset
    )
    return ServiceResponse.build_success_response(data=threads)


@router.delete("/{thread_id}")
async def delete_thread(
    user_id: Annotated[int, Query(gt=0)],
    thread_id: Annotated[str, Path(min_length=1, max_length=128)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    await thread_service.delete_thread(db, user_id=user_id, thread_id=thread_id)

    return ServiceResponse.build_success_response(
        data={"thread_id": thread_id, "deleted": True}
    )
