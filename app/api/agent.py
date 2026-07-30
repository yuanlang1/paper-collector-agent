from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.schemas.agent import (
    ChatRequest,
    ChatResponse,
    ChatResumeRequest,
)
from app.core.response import ServiceResponse
from app.database import get_db
from app.llm.agent import get_agent_service


router = APIRouter(
    prefix="/api/agent",
    tags=["agent"],
)


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def generate_conversation_id() -> str:
    return f"conv_{uuid4().hex}"


def resolve_conversation_id(
    conversation_id: str | None,
) -> str:
    if conversation_id:
        return conversation_id

    return generate_conversation_id()


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
    conversation_id = resolve_conversation_id(
        payload.conversation_id
    )

    service = get_agent_service()

    result = await service.chat(
        message=payload.message,
        conversation_id=conversation_id,
        forced_subagent=(
            {
                "name": payload.subagent.name,
                "input": {
                    "prompt": payload.message,
                    "constraints": payload.subagent.constraints.model_dump(
                        mode="json",
                        exclude_none=True,
                    ),
                },
            }
            if payload.subagent is not None
            else None
        ),
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
    conversation_id = resolve_conversation_id(
        payload.conversation_id
    )

    service = get_agent_service()

    return build_streaming_response(
        service.chat_stream(
            message=payload.message,
            conversation_id=conversation_id,
            forced_subagent=(
                {
                    "name": payload.subagent.name,
                    "input": {
                        "prompt": payload.message,
                        "constraints": payload.subagent.constraints.model_dump(
                            mode="json",
                            exclude_none=True,
                        ),
                    },
                }
                if payload.subagent is not None
                else None
            ),
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
    service = get_agent_service()

    result = await service.resume_chat(
        conversation_id=payload.conversation_id,
        resume_payload=payload.model_dump(
            exclude={"conversation_id"},
            exclude_none=True,
        ),
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
    service = get_agent_service()

    return build_streaming_response(
        service.resume_chat_stream(
            conversation_id=payload.conversation_id,
            resume_payload=payload.model_dump(
                exclude={"conversation_id"},
                exclude_none=True,
            ),
            db=db,
        )
    )
