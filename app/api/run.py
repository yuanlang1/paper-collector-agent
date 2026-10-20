import json
from contextlib import aclosing
from typing import Annotated, Any

from anyio import CancelScope
from fastapi import APIRouter, Depends, Path, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.response import ServiceResponse
from app.database import get_db
from app.db.models import AgentRun, RunStatus
from app.schemas.ai import ChatRequest, RunInfo
from app.services import agent_run_service, thread_service

router = APIRouter(prefix="/api/agent", tags=["agent"])


def _sse_line(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class RunStreamingResponse(StreamingResponse):
    def __init__(
        self, run: AgentRun, session_factory, checkpointer, store, *, input: dict[str, Any],
    ):
        self.run_record = run
        self.run_input = input
        self.session_factory = session_factory
        super().__init__(
            self._events(checkpointer, store),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _events(self, checkpointer, store):
        async with (
            self.session_factory() as db,
            aclosing(
                agent_run_service.run(
                    db=db,
                    run=self.run_record,
                    input=self.run_input,
                    checkpointer=checkpointer,
                    store=store,
                )
            ) as events,
        ):
            async for event in events:
                yield _sse_line(event)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with CancelScope(shield=True):
                try:
                    await self.body_iterator.aclose()
                finally:
                    async with self.session_factory() as db:
                        await agent_run_service.update_status(
                            db,
                            user_id=self.run_record.user_id,
                            thread_id=self.run_record.thread_id,
                            run_id=self.run_record.run_id,
                            status=RunStatus.CANCELLED,
                        )


@router.post("/run")
async def chat(
    body: ChatRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    run = await agent_run_service.create_run(
        db,
        user_id=body.user_id,
        thread_id=body.thread_id,
        metadata=body.metadata.model_dump(exclude_unset=True),
    )
    return RunStreamingResponse(
        run,
        request.app.state.session_factory,
        request.app.state.checkpointer,
        request.app.state.store,
        input=body.model_dump(
            include={"message"} if body.message is not None else {"resume"},
            exclude_unset=True,
        ),
    )


@router.get("/threads/{thread_id}/runs")
async def list_runs(
    user_id: Annotated[int, Query(gt=0)],
    thread_id: Annotated[str, Path(min_length=1, max_length=128)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    runs = await agent_run_service.list_runs(
        db,
        user_id=user_id,
        thread_id=thread_id,
        limit=limit,
        offset=offset,
    )
    return ServiceResponse.build_success_response(
        data=[RunInfo.model_validate(run) for run in runs],
    )


@router.get("/threads/{thread_id}/runs/{run_id}")
async def get_run(
    user_id: Annotated[int, Query(gt=0)],
    thread_id: Annotated[str, Path(min_length=1, max_length=128)],
    run_id: Annotated[str, Path(min_length=1, max_length=128)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    run = await agent_run_service.get_run(
        db,
        user_id=user_id,
        thread_id=thread_id,
        run_id=run_id,
    )
    return ServiceResponse.build_success_response(data=RunInfo.model_validate(run))


@router.get("/history")
async def get_history(
    user_id: Annotated[int, Query(gt=0)],
    thread_id: Annotated[str, Query(min_length=1)],
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    history = await thread_service.get_history(
        db,
        user_id=user_id,
        thread_id=thread_id,
        checkpointer=request.app.state.checkpointer,
    )

    return ServiceResponse.build_success_response(data=history)
