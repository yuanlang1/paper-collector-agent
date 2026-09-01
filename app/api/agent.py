from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, Query
from app.api.schemas.chat_history import (
    ChatMessageListResponse,
    ChatMessageView,
    ConversationListResponse,
    ConversationView,
)
from app.history.store import get_history_store

from app.api.schemas.agent import (
    ChatRequest,
    ChatResponse,
    ChatResumeRequest,
)
from app.core.response import ServiceResponse
from app.database import get_db
from app.runtime import get_agent_runtime


router = APIRouter(
    prefix="/api/agent",
    tags=["agent"],
)


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def build_streaming_response(
    stream: AsyncIterator[str],
) -> StreamingResponse:
    return StreamingResponse(
        content=stream,
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post(
    "/chat",
    response_model=ServiceResponse[ChatResponse],
)
async def agent_chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
) -> ServiceResponse[ChatResponse]:
    runtime = get_agent_runtime()

    result = await runtime.chat(
        message=payload.message,
        conversation_id=payload.conversation_id,
        db=db,
    )

    response = ChatResponse.model_validate(result)

    return ServiceResponse[
        ChatResponse
    ].build_success_response(
        data=response,
        message="OK",
    )


@router.post("/chat/stream")
async def stream_chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    runtime = get_agent_runtime()

    return build_streaming_response(
        runtime.chat_stream(
            message=payload.message,
            conversation_id=payload.conversation_id,
            db=db,
        )
    )


@router.post(
    "/chat/resume",
    response_model=ServiceResponse[ChatResponse],
)
async def chat_resume(
    payload: ChatResumeRequest,
    db: Session = Depends(get_db),
) -> ServiceResponse[ChatResponse]:
    runtime = get_agent_runtime()

    result = await runtime.resume_chat(
        conversation_id=payload.conversation_id,
        resume_payload=payload.model_dump(
            exclude={"conversation_id", "run_id", "action_id"},
            exclude_none=True,
        ),
        requested_run_id=payload.run_id,
        requested_action_id=payload.action_id,
        db=db,
    )

    response = ChatResponse.model_validate(result)

    return ServiceResponse[
        ChatResponse
    ].build_success_response(
        data=response,
        message="OK",
    )


@router.post("/chat/resume/stream")
async def chat_resume_stream(
    payload: ChatResumeRequest,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    runtime = get_agent_runtime()

    return build_streaming_response(
        runtime.resume_chat_stream(
            conversation_id=payload.conversation_id,
            resume_payload=payload.model_dump(
                exclude={"conversation_id", "run_id", "action_id"},
                exclude_none=True,
            ),
            requested_run_id=payload.run_id,
            requested_action_id=payload.action_id,
            db=db,
        )
    )

@router.get(
    "/conversations",
    response_model=ServiceResponse[ConversationListResponse],
)
async def list_conversations(
    limit: int = Query(default=30, ge=1, le=100),
) -> ServiceResponse[ConversationListResponse]:
    history_store = get_history_store()
    items = await history_store.list_conversations(limit=limit)

    return ServiceResponse[ConversationListResponse].build_success_response(
        data=ConversationListResponse(
            items=[
                ConversationView.model_validate(item)
                for item in items
            ]
        ),
        message="OK",
    )


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=ServiceResponse[ChatMessageListResponse],
)
async def list_conversation_messages(
    conversation_id: str,
    limit: int = Query(default=100, ge=1, le=100),
    before_id: int | None = Query(default=None, ge=1),
) -> ServiceResponse[ChatMessageListResponse]:
    history_store = get_history_store()

    items, next_before_id = await history_store.list_messages(
        conversation_id=conversation_id,
        limit=limit,
        before_id=before_id,
    )

    return ServiceResponse[ChatMessageListResponse].build_success_response(
        data=ChatMessageListResponse(
            conversation_id=conversation_id,
            items=[
                ChatMessageView.model_validate(item)
                for item in items
            ],
            next_before_id=next_before_id,
        ),
        message="OK",
    )
