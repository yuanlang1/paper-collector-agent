from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from sqlalchemy.orm import Session
from app.api.schemas.chat_history import (
    ChatMessageListResponse,
    ChatMessageView,
    ConversationListResponse,
    ConversationView,
    DeleteConversationResponse,
)
from app.history import chat_log
from app.history.sqlite import get_history_database

from app.api.schemas.agent import (
    ChatRequest,
    ChatResponse,
    ChatResumeRequest,
)
from app.core.response import ServiceResponse
from app.database import get_db
from app.runtime import get_agent_runtime
from app.runtime.agent_runtime import AgentStreamSubscription


router = APIRouter(
    prefix="/api/agent",
    tags=["agent"],
)


async def _start_chat_stream(
    payload: ChatRequest,
    response: Response,
    db: Session = Depends(get_db),
) -> AgentStreamSubscription:
    try:
        subscription = await get_agent_runtime().chat_stream(
            message=payload.message,
            user_id=payload.user_id,
            conversation_id=payload.conversation_id,
            run_id=payload.run_id,
            llm_profile_id=payload.llm_profile_id,
            db=db,
        )
    except chat_log.RunAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    response.headers["X-Agent-Run-ID"] = subscription.run_id
    response.headers["Access-Control-Expose-Headers"] = "X-Agent-Run-ID"
    return subscription


async def _start_resume_stream(
    payload: ChatResumeRequest,
    response: Response,
    db: Session = Depends(get_db),
) -> AgentStreamSubscription:
    subscription = await get_agent_runtime().resume_chat_stream(
        user_id=payload.user_id,
        conversation_id=payload.conversation_id,
        resume_payload=payload.model_dump(
            exclude={"user_id", "conversation_id", "run_id", "action_id"},
            exclude_none=True,
        ),
        requested_run_id=payload.run_id,
        requested_action_id=payload.action_id,
        db=db,
    )
    response.headers["X-Agent-Run-ID"] = subscription.run_id
    response.headers["Access-Control-Expose-Headers"] = "X-Agent-Run-ID"
    return subscription


async def _subscribe_to_run(
    run_id: str,
    user_id: Annotated[str, Query(min_length=1)],
    conversation_id: Annotated[str, Query(min_length=1)],
    last_event_id: Annotated[str | None, Header()] = None,
) -> AgentStreamSubscription:
    cursor: int | None = None
    if last_event_id is not None:
        if not last_event_id.isdecimal() or int(last_event_id) < 1:
            raise HTTPException(status_code=400, detail="Invalid Last-Event-ID")
        cursor = int(last_event_id)
    try:
        return await get_agent_runtime().subscribe_stream(
            user_id=user_id,
            conversation_id=conversation_id,
            run_id=run_id,
            last_event_sequence=cursor,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _sse_events(
    subscription: AgentStreamSubscription,
) -> AsyncIterator[ServerSentEvent]:
    async for event in subscription.events:
        yield ServerSentEvent(
            id=str(event.sequence),
            event=event.event_name,
            data=event.data,
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

    try:
        result = await runtime.chat(
            message=payload.message,
            user_id=payload.user_id,
            conversation_id=payload.conversation_id,
            run_id=payload.run_id,
            llm_profile_id=payload.llm_profile_id,
            db=db,
        )
    except chat_log.RunAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    response = ChatResponse.model_validate(result)

    return ServiceResponse[
        ChatResponse
    ].build_success_response(
        data=response,
        message="OK",
    )


@router.post("/chat/stream", response_class=EventSourceResponse)
async def stream_chat(
    subscription: Annotated[AgentStreamSubscription, Depends(_start_chat_stream)],
) -> AsyncIterator[ServerSentEvent]:
    async for event in _sse_events(subscription):
        yield event


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
        user_id=payload.user_id,
        conversation_id=payload.conversation_id,
        resume_payload=payload.model_dump(
            exclude={"user_id", "conversation_id", "run_id", "action_id"},
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


@router.post("/chat/resume/stream", response_class=EventSourceResponse)
async def chat_resume_stream(
    subscription: Annotated[AgentStreamSubscription, Depends(_start_resume_stream)],
) -> AsyncIterator[ServerSentEvent]:
    async for event in _sse_events(subscription):
        yield event


@router.get("/runs/{run_id}/events", response_class=EventSourceResponse)
async def stream_run_events(
    subscription: Annotated[AgentStreamSubscription, Depends(_subscribe_to_run)],
) -> AsyncIterator[ServerSentEvent]:
    async for event in _sse_events(subscription):
        yield event

@router.get(
    "/conversations",
    response_model=ServiceResponse[ConversationListResponse],
)
async def list_conversations(
    limit: int = Query(default=30, ge=1, le=100),
) -> ServiceResponse[ConversationListResponse]:
    history_db = get_history_database()
    items = await history_db.run(
        chat_log.list_conversations,
        user_id="0",
        limit=limit,
    )

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
    history_db = get_history_database()
    items, next_before_id = await history_db.run(
        chat_log.list_messages,
        user_id="0",
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


@router.delete(
    "/conversations/{conversation_id}",
    response_model=ServiceResponse[DeleteConversationResponse],
)
async def delete_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
) -> ServiceResponse[DeleteConversationResponse]:
    runtime = get_agent_runtime()
    result = await runtime.delete_conversation(
        conversation_id=conversation_id,
        db=db,
    )

    return ServiceResponse[DeleteConversationResponse].build_success_response(
        data=DeleteConversationResponse.model_validate(result),
        message="OK",
    )
