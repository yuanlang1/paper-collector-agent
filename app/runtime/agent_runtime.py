import time

from app.history.store import ChatHistoryStore
from collections.abc import AsyncIterator
from typing import Any, Optional
from uuid import uuid4
from sqlalchemy.orm import Session as DbSession
from app.llm.agent import AgentService
from app.runtime.session import Session


class AgentRuntime:
    def __init__(
        self,
        agent_service: AgentService,
        history_store: ChatHistoryStore,
    ) -> None:
        self.agent_service = agent_service
        self.history_store = history_store

    async def chat(
        self,
        *,
        message: str,
        conversation_id: str | None,
        db: DbSession,
    ) -> dict[str, Any]:
        session = Session.create(
            message=message,
            conversation_id=self.resolve_conversation_id(conversation_id),
            db=db,
        )

        assistant_message_id = await self.history_store.start_turn(
            conversation_id=session.conversation_id,
            run_id=str(session.run_id),
            user_content=message,
        )
        started_at = time.perf_counter()

        try:
            response = await self.agent_service.invoke(session)
        except Exception:
            await self.history_store.mark_interrupted(
                message_id=assistant_message_id,
            )
            raise

        await self.history_store.complete_assistant_message(
            message_id=assistant_message_id,
            response=response,
            latency_ms=self._elapsed_ms(started_at),
        )

        return response

    async def chat_stream(
        self,
        *,
        message: str,
        conversation_id: str | None,
        db: DbSession,
    ) -> AsyncIterator[str]:
        session = Session.create(
            message=message,
            conversation_id=self.resolve_conversation_id(conversation_id),
            db=db,
        )

        assistant_message_id = await self.history_store.start_turn(
            conversation_id=session.conversation_id,
            run_id=str(session.run_id),
            user_content=message,
        )
        started_at = time.perf_counter()
        terminal_persisted = False

        async def persist_terminal(response: dict[str, Any]) -> None:
            nonlocal terminal_persisted

            if terminal_persisted:
                return

            await self.history_store.complete_assistant_message(
                message_id=assistant_message_id,
                response=response,
                latency_ms=self._elapsed_ms(started_at),
            )
            terminal_persisted = True

        try:
            async for event in self.agent_service.stream(
                session,
                on_terminal=persist_terminal,
            ):
                yield event
        finally:
            if not terminal_persisted:
                await self.history_store.mark_interrupted(
                    message_id=assistant_message_id,
                )

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
        assistant_message_id = await self.history_store.start_resume(
            conversation_id=conversation_id,
            run_id=str(session.run_id),
            source="resume",
        )
        started_at = time.perf_counter()

        try:
            response = await self.agent_service.invoke(session)
        except Exception:
            await self.history_store.mark_interrupted(
                message_id=assistant_message_id,
            )
            raise

        await self.history_store.complete_assistant_message(
            message_id=assistant_message_id,
            response=response,
            latency_ms=self._elapsed_ms(started_at),
            extra_meta=self._resume_meta(resume_payload),
        )

        return response

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
        assistant_message_id = await self.history_store.start_resume(
            conversation_id=conversation_id,
            run_id=str(session.run_id),
            source="resume",
        )
        started_at = time.perf_counter()
        terminal_persisted = False

        async def persist_terminal(response: dict[str, Any]) -> None:
            nonlocal terminal_persisted

            if terminal_persisted:
                return

            await self.history_store.complete_assistant_message(
                message_id=assistant_message_id,
                response=response,
                latency_ms=self._elapsed_ms(started_at),
                extra_meta=self._resume_meta(resume_payload),
            )
            terminal_persisted = True

        try:
            async for event in self.agent_service.stream(
                session,
                on_terminal=persist_terminal,
            ):
                yield event
        finally:
            if not terminal_persisted:
                await self.history_store.mark_interrupted(
                    message_id=assistant_message_id,
                )

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

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return int((time.perf_counter() - started_at) * 1000)

    @staticmethod
    def _resume_meta(
        resume_payload: dict[str, Any],
    ) -> dict[str, Any]:
        resume = {
            "decision": resume_payload.get("decision"),
            "comment": resume_payload.get("comment"),
            "has_query_understanding_override": (
                resume_payload.get("query_understanding") is not None
            ),
            "has_search_tag_override": (
                resume_payload.get("search_tag") is not None
            ),
        }

        return {
            "resume": {
                key: value
                for key, value in resume.items()
                if value not in (None, "")
            }
        }


_agent_runtime: AgentRuntime | None = None

def initialize_agent_runtime(
    checkpointer,
    history_store: ChatHistoryStore,
) -> None:
    global _agent_runtime

    _agent_runtime = AgentRuntime(
        agent_service=AgentService(checkpointer=checkpointer),
        history_store=history_store,
    )

def get_agent_runtime() -> AgentRuntime:
    if _agent_runtime is None:
        raise RuntimeError("Agent runtime has not been initialized")
    return _agent_runtime
