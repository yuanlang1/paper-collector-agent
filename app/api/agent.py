import json
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.agent import AgentChatRequest
from app.db.session import get_db
from app.services import agent_service


router = APIRouter(prefix="/api/agent", tags=["agent"])


def _sse_line(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(
    body: AgentChatRequest, 
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)]
):
    async def event_gen():
        async for event in agent_service.chat(
            db = db,
            conversation_id = body.conversation_id,
            user_id = "0",
            message = body.message,
            checkpointer = request.app.state.checkpointer,
            store = request.app.state.store
        ):
            yield _sse_line(event)

    return StreamingResponse(
        event_gen(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )