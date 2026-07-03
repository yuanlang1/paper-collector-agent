from typing import Dict, List, Literal, Optional, Any
from uuid import uuid4
import uuid
from openai import conversations
from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.response import ServiceResponse
from app.database import get_db
from app.llm.langchain_service import get_agent_service
from fastapi.responses import StreamingResponse

from app.llm.tools.registry import requires_confirmation

router = APIRouter(prefix="/api/agent", tags=["agent"])

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., description="消息内容")

class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    conversation_id: Optional[str]= Field(default=None, description="对话ID，用于 LangGraph thread_id")

class AgentAction(BaseModel):
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    success: bool = True
    error: str | None = None

class ChatResumeRequest(BaseModel):
    conversation_id: str = Field(..., description="要恢复的对话 ID / LangGraph thread_id")
    approved: bool = Field(..., description="是否批准继续执行")
    comment: Optional[str] = Field(default=None, description="人工确认备注")

class ChatResponse(BaseModel):
    conversation_id: str
    status: Literal[
        "completed",
        "confirmation_required",
        "tool_executed",
        "failed",
        "rejected",
    ]

    route: Literal["ask", "agent"] | None = None
    reply: str

    requires_confirmation: bool = False
    reasoning: str | None = None

    actions: list[AgentAction] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

@router.post("/chat")
async def agent_chat(
    payload: ChatRequest,
    db: Session = Depends(get_db)
) -> ServiceResponse[ChatResponse]:
    conversation_id = (payload.conversation_id or "").strip()
    if not conversation_id:
        conversation_id = f"conv-{uuid4().hex}"

    service = get_agent_service()

    result = await service.chat(
        message = payload.message,
        conversation_id = conversation_id,
        db = db,
    )

    return ServiceResponse.build_success_response(
        ChatResponse(**result),
        message = "OK",
    )

@router.post("/chat/stream")
async def stream_chat(
    payload: ChatRequest,
    db: Session = Depends(get_db)
) -> StreamingResponse:
    conversation_id = (payload.conversation_id or "").strip()

    if not conversation_id:
        conversation_id = f"conv_{uuid4().hex}"

    service = get_agent_service()

    return StreamingResponse(
        service.chat_stream(
            message = payload.message,
            conversation_id = conversation_id,
            db = db,
        ),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@router.post("/chat/resume")
async def chat_resume(
    payload: ChatResumeRequest,
    db: Session = Depends(get_db)
) -> ServiceResponse[ChatResponse]:
    conversation_id = payload.conversation_id.strip()

    if not conversation_id:
        raise ValueError("conversation_id 不能为空")

    service = get_agent_service()

    result = await service.resume_chat(
        conversation_id = conversation_id,
        resume_payload = {
            "approved": payload.approved,
            "comment": payload.comment,
        },
        db = db,
    )

    return ServiceResponse.build_success_response(
        ChatResponse(**result),
        message = "OK",
    )
