from collections.abc import AsyncIterator
from typing import Any, Optional
from uuid import uuid4

from sqlalchemy.orm import Session as DbSession

from app.llm.agent import AgentService
from app.runtime.session import Session


class AgentRuntime:
    def __init__(
        self,
        agent_service: AgentService | None = None,
    ) -> None:
        self.agent_service = agent_service or AgentService()

    async def chat(
        self,
        *,
        message: str,
        conversation_id: str | None,
        db: DbSession,
    ) -> dict[str, Any]:
        session = Session.create(
            message=message,
            conversation_id=self.resolve_conversation_id(
                conversation_id,
            ),
            db=db,
        )
        return await self.agent_service.invoke(session)

    async def chat_stream(
        self,
        *,
        message: str,
        conversation_id: str | None,
        db: DbSession,
    ) -> AsyncIterator[str]:
        session = Session.create(
            message=message,
            conversation_id=self.resolve_conversation_id(
                conversation_id,
            ),
            db=db,
        )
        async for event in self.agent_service.stream(session):
            yield event

    async def resume_chat(
        self,
        *,
        conversation_id: str,
        resume_payload: dict[str, Any],
        db: DbSession,
    ) -> dict[str, Any]:
        session = await self._resume_session(
            conversation_id=conversation_id,
            resume_payload=resume_payload,
            db=db,
        )
        return await self.agent_service.invoke(session)

    async def resume_chat_stream(
        self,
        *,
        conversation_id: str,
        resume_payload: dict[str, Any],
        db: DbSession,
    ) -> AsyncIterator[str]:
        session = await self._resume_session(
            conversation_id=conversation_id,
            resume_payload=resume_payload,
            db=db,
        )
        async for event in self.agent_service.stream(session):
            yield event

    async def _resume_session(
        self,
        *,
        conversation_id: str,
        resume_payload: dict[str, Any],
        db: DbSession,
    ) -> Session:
        lookup = Session.for_resume_lookup(
            conversation_id=conversation_id,
            db=db,
        )
        snapshot = await self.agent_service.get_state(lookup)

        if not snapshot.values or not snapshot.next:
            raise ValueError("当前会话没有等待恢复的运行")

        return Session.resume(
            conversation_id=conversation_id,
            run_id=str(snapshot.values["run_id"]),
            resume_payload=resume_payload,
            db=db,
        )

    @staticmethod
    def resolve_conversation_id(
        conversation_id: str | None,
    ) -> str:
        if conversation_id:
            return conversation_id

        return f"conv_{uuid4().hex}"


_agent_runtime: Optional[AgentRuntime] = None


def get_agent_runtime() -> AgentRuntime:
    global _agent_runtime

    if _agent_runtime is None:
        _agent_runtime = AgentRuntime()

    return _agent_runtime
