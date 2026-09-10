import asyncio
import logging
import time

from sqlalchemy import delete, select
from app.database import SessionLocal
from app.core.exceptions import ConflictException
from app.history.store import ChatHistoryStore
from app.llm.model_factory import use_llm_runtime_config
from app.memory.consolidation import Consolidator
from app.memory.extraction import LangChainMemoryExtractor
from app.rag.index_construction.memory_index_sync import (
    get_memory_index_synchronizer,
)
from app.models.memory import (
    MemoryConsolidationCursor,
    MemoryEpisode,
    MemoryFact,
)
from collections.abc import AsyncIterator, Awaitable
from typing import Any, Callable, Optional
from uuid import uuid4
from sqlalchemy.orm import Session as DbSession
from app.llm.agent import AgentService
from app.runtime.session import Session
from app.runtime.threaded_stream import DetachedStreamRun
from app.services.llm_profile_service import (
    LlmRuntimeConfig,
    resolve_runtime_config,
    resolve_small_model_runtime_config,
)
from app.services.setting_service import get_source_limits


logger = logging.getLogger(__name__)
DEFAULT_USER_ID = "0"


class AgentRuntime:
    def __init__(
        self,
        agent_service: AgentService,
        history_store: ChatHistoryStore,
        db_factory: Callable[[], DbSession] = SessionLocal,
    ) -> None:
        self.agent_service = agent_service
        self.history_store = history_store
        self.db_factory = db_factory
        self._active_stream_runs: set[DetachedStreamRun] = set()
        self._active_stream_tasks: set[asyncio.Task[None]] = set()
        self._active_consolidation_scopes: set[tuple[str, str]] = set()
        self._active_consolidation_tasks: set[asyncio.Task[None]] = set()
        self._active_conversation_runs: dict[str, int] = {}
        self._deleting_conversations: set[str] = set()
        
    async def chat(
        self,
        *,
        message: str,
        conversation_id: str | None,
        db: DbSession,
        llm_profile_id: int | None = None,
    ) -> dict[str, Any]:
        llm_config = resolve_runtime_config(db, llm_profile_id)
        memory_llm_config = resolve_small_model_runtime_config(db)
        resolved_conversation_id = self.resolve_conversation_id(conversation_id)
        source_limits = get_source_limits(db)
        self._begin_conversation_run(resolved_conversation_id)
        try:
            session = Session.create(
                message=message,
                conversation_id=resolved_conversation_id,
                db=db,
                user_id=DEFAULT_USER_ID,
                llm_profile=llm_config.snapshot() if llm_config else None,
                memory_llm_profile=self._memory_llm_profile_snapshot(
                    llm_config,
                    memory_llm_config,
                ),
                paper_search_source_limits=source_limits,
            )

            assistant_message_id = await self.history_store.start_turn(
                user_id=session.user_id,
                conversation_id=session.conversation_id,
                run_id=str(session.run_id),
                user_content=message,
            )
            started_at = time.perf_counter()

            try:
                response = await self._service_for_config(
                    llm_config,
                    memory_llm_config,
                ).invoke(session)
            except Exception:
                await self.history_store.mark_interrupted(
                    message_id=assistant_message_id,
                )
                raise

            await self.history_store.complete_assistant_message(
                message_id=assistant_message_id,
                response=response,
                latency_ms=self._elapsed_ms(started_at),
                extra_meta=self._llm_profile_meta(session),
            )
            self._schedule_consolidation(session=session, response=response)

            return response
        finally:
            self._finish_conversation_run(resolved_conversation_id)

    async def chat_stream(
        self,
        *,
        message: str,
        conversation_id: str | None,
        db: DbSession,
        llm_profile_id: int | None = None,
    ) -> AsyncIterator[str]:
        llm_config = resolve_runtime_config(db, llm_profile_id)
        memory_llm_config = resolve_small_model_runtime_config(db)
        resolved_conversation_id = self.resolve_conversation_id(conversation_id)
        source_limits = get_source_limits(db)
        run_id = f"run_{uuid4().hex}"
        self._begin_conversation_run(resolved_conversation_id)
        try:
            assistant_message_id = await self.history_store.start_turn(
                user_id=DEFAULT_USER_ID,
                conversation_id=resolved_conversation_id,
                run_id=run_id,
                user_content=message,
            )

            stream_run = DetachedStreamRun(run_id)
            self._register_stream_run(stream_run)
            self._start_stream_task(
                self._run_stream_task(
                    stream_run=stream_run,
                    message=message,
                    conversation_id=resolved_conversation_id,
                    run_id=run_id,
                    resume_payload=None,
                    resume_action_id=None,
                    assistant_message_id=assistant_message_id,
                    service=self._service_for_config(
                        llm_config,
                        memory_llm_config,
                    ),
                    llm_profile=llm_config.snapshot() if llm_config else None,
                    memory_llm_profile=self._memory_llm_profile_snapshot(
                        llm_config,
                        memory_llm_config,
                    ),
                    paper_search_source_limits=source_limits,
                ),
                run_id=run_id,
            )
        except Exception:
            self._finish_conversation_run(resolved_conversation_id)
            raise

        return stream_run.subscribe()

    async def resume_chat(
        self,
        *,
        conversation_id: str,
        resume_payload: dict[str, Any],
        requested_run_id: str | None = None,
        requested_action_id: str | None = None,
        db: DbSession,
    ) -> dict[str, Any]:
        self._begin_conversation_run(conversation_id)
        try:
            session, action_id, service = await self._resume_session(
                conversation_id=conversation_id,
                resume_payload=resume_payload,
                requested_run_id=requested_run_id,
                requested_action_id=requested_action_id,
                db=db,
            )
            claim = await self.history_store.claim_pending_action(
                user_id=session.user_id,
                conversation_id=conversation_id,
                run_id=str(session.run_id),
                action_id=action_id,
                decision=str(resume_payload["decision"]),
                comment=resume_payload.get("comment"),
            )
            if not claim.claimed:
                raise ValueError("The requested approval is already being processed")
            assistant_message_id = claim.message_id
            started_at = time.perf_counter()

            try:
                response = await service.invoke(session)
            except Exception:
                await self.history_store.mark_interrupted(
                    message_id=assistant_message_id,
                )
                raise

            await self.history_store.complete_assistant_message(
                message_id=assistant_message_id,
                response=response,
                latency_ms=self._elapsed_ms(started_at),
                extra_meta={
                    **self._resume_meta(resume_payload, action_id),
                    **self._llm_profile_meta(session),
                },
            )
            self._schedule_consolidation(session=session, response=response)

            return response
        finally:
            self._finish_conversation_run(conversation_id)

    async def resume_chat_stream(
        self,
        *,
        conversation_id: str,
        resume_payload: dict[str, Any],
        requested_run_id: str | None = None,
        requested_action_id: str | None = None,
        db: DbSession,
    ) -> AsyncIterator[str]:
        self._begin_conversation_run(conversation_id)
        try:
            session, action_id, service = await self._resume_session(
                conversation_id=conversation_id,
                resume_payload=resume_payload,
                requested_run_id=requested_run_id,
                requested_action_id=requested_action_id,
                db=db,
            )
            run_id = str(session.run_id)
            claim = await self.history_store.claim_pending_action(
                user_id=session.user_id,
                conversation_id=conversation_id,
                run_id=run_id,
                action_id=action_id,
                decision=str(resume_payload["decision"]),
                comment=resume_payload.get("comment"),
            )
            if not claim.claimed:
                raise ValueError("The requested approval is already being processed")
            assistant_message_id = claim.message_id

            stream_run = DetachedStreamRun(run_id)
            self._register_stream_run(stream_run)
            self._start_stream_task(
                self._run_stream_task(
                    stream_run=stream_run,
                    message=None,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    resume_payload=resume_payload,
                    resume_action_id=action_id,
                    assistant_message_id=assistant_message_id,
                    service=service,
                    llm_profile=session.llm_profile,
                    memory_llm_profile=session.memory_llm_profile,
                    paper_search_source_limits=None,
                ),
                run_id=run_id,
            )
        except Exception:
            self._finish_conversation_run(conversation_id)
            raise

        return stream_run.subscribe()

    async def delete_conversation(
        self,
        *,
        conversation_id: str,
        db: DbSession,
    ) -> dict[str, Any]:
        if self._is_conversation_busy(conversation_id):
            raise ConflictException("会话正在执行，暂不能删除")
        if conversation_id in self._deleting_conversations:
            raise ConflictException("会话正在删除，请稍后重试")

        self._deleting_conversations.add(conversation_id)
        try:
            await self.agent_service.checkpointer.adelete_thread(conversation_id)
            await self._delete_conversation_memory(
                db=db,
                conversation_id=conversation_id,
            )
            await self.history_store.delete_conversation(
                user_id=DEFAULT_USER_ID,
                conversation_id=conversation_id,
            )
        finally:
            self._deleting_conversations.discard(conversation_id)

        return {
            "conversation_id": conversation_id,
            "deleted": True,
        }

    def _register_stream_run(self, stream_run: DetachedStreamRun) -> None:
        self._active_stream_runs.add(stream_run)

    def _unregister_stream_run(self, stream_run: DetachedStreamRun) -> None:
        self._active_stream_runs.discard(stream_run)

    def _start_stream_task(
        self,
        coroutine: Awaitable[None],
        *,
        run_id: str,
    ) -> None:
        task = asyncio.create_task(coroutine, name=f"agent-stream:{run_id}")
        self._active_stream_tasks.add(task)
        task.add_done_callback(self._active_stream_tasks.discard)

    def _begin_conversation_run(self, conversation_id: str) -> None:
        if conversation_id in self._deleting_conversations:
            raise ConflictException("会话正在删除，暂不能执行")
        self._active_conversation_runs[conversation_id] = (
            self._active_conversation_runs.get(conversation_id, 0) + 1
        )

    def _finish_conversation_run(self, conversation_id: str) -> None:
        active_count = self._active_conversation_runs.get(conversation_id, 0)
        if active_count <= 1:
            self._active_conversation_runs.pop(conversation_id, None)
            return
        self._active_conversation_runs[conversation_id] = active_count - 1

    def _is_conversation_busy(self, conversation_id: str) -> bool:
        return (
            self._active_conversation_runs.get(conversation_id, 0) > 0
            or (DEFAULT_USER_ID, conversation_id)
            in self._active_consolidation_scopes
        )

    @staticmethod
    async def _delete_conversation_memory(
        *,
        db: DbSession,
        conversation_id: str,
    ) -> None:
        try:
            fact_ids = list(
                db.scalars(
                    select(MemoryFact.id).where(
                        MemoryFact.user_id == DEFAULT_USER_ID,
                        MemoryFact.source_conversation_id == conversation_id,
                    )
                )
            )
            episode_ids = list(
                db.scalars(
                    select(MemoryEpisode.id).where(
                        MemoryEpisode.user_id == DEFAULT_USER_ID,
                        MemoryEpisode.source_conversation_id == conversation_id,
                    )
                )
            )
            db.execute(
                delete(MemoryFact).where(
                    MemoryFact.user_id == DEFAULT_USER_ID,
                    MemoryFact.source_conversation_id == conversation_id,
                )
            )
            db.execute(
                delete(MemoryEpisode).where(
                    MemoryEpisode.user_id == DEFAULT_USER_ID,
                    MemoryEpisode.source_conversation_id == conversation_id,
                )
            )
            db.execute(
                delete(MemoryConsolidationCursor).where(
                    MemoryConsolidationCursor.user_id == DEFAULT_USER_ID,
                    MemoryConsolidationCursor.conversation_id == conversation_id,
                )
            )
            db.commit()
            synchronizer = get_memory_index_synchronizer()
            await synchronizer.delete_facts(fact_ids)
            await synchronizer.delete_episodes(episode_ids)
        except Exception:
            db.rollback()
            raise

    def _schedule_consolidation(
        self,
        *,
        session: Session,
        response: dict[str, Any],
    ) -> None:
        if response.get("status") != "completed":
            return

        scope = (session.user_id, session.conversation_id)
        if scope in self._active_consolidation_scopes:
            return

        self._active_consolidation_scopes.add(scope)
        task = asyncio.create_task(
            self._run_consolidation(
                user_id=session.user_id,
                conversation_id=session.conversation_id,
                memory_llm_profile=session.memory_llm_profile,
            ),
            name=(
                "memory-consolidation:"
                f"{session.user_id}:{session.conversation_id}"
            ),
        )
        self._active_consolidation_tasks.add(task)

        def cleanup(completed_task: asyncio.Task[None]) -> None:
            self._active_consolidation_tasks.discard(completed_task)
            self._active_consolidation_scopes.discard(scope)

        task.add_done_callback(cleanup)

    async def _run_consolidation(
        self,
        *,
        user_id: str,
        conversation_id: str,
        memory_llm_profile: dict[str, Any] | None,
    ) -> None:
        db = self.db_factory()
        try:
            profile_id = (
                memory_llm_profile.get("profile_id")
                if isinstance(memory_llm_profile, dict)
                else None
            )
            memory_llm_config = resolve_runtime_config(db, profile_id)
            with use_llm_runtime_config(memory_llm_config):
                result = await Consolidator(
                    db,
                    user_id=user_id,
                    history_reader=self.history_store,
                    extraction_model=LangChainMemoryExtractor(),
                ).consolidate_if_due(conversation_id=conversation_id)
            if result.due:
                logger.info(
                    "Memory consolidated: user_id=%s, conversation_id=%s, "
                    "facts_created=%s, episode_created=%s",
                    user_id,
                    conversation_id,
                    result.facts_created,
                    result.episode_created,
                )
        except Exception:
            logger.exception(
                "Memory consolidation failed: user_id=%s, conversation_id=%s",
                user_id,
                conversation_id,
            )
        finally:
            db.close()

    async def _run_stream_task(
        self,
        *,
        stream_run: DetachedStreamRun,
        message: str | None,
        conversation_id: str,
        run_id: str,
        resume_payload: dict[str, Any] | None,
        resume_action_id: str | None,
        assistant_message_id: int,
        service: AgentService,
        llm_profile: dict[str, Any] | None,
        memory_llm_profile: dict[str, Any] | None,
        paper_search_source_limits: dict[str, int] | None,
    ) -> None:
        try:
            await self._run_stream_worker(
                stream_run=stream_run,
                message=message,
                conversation_id=conversation_id,
                run_id=run_id,
                resume_payload=resume_payload,
                resume_action_id=resume_action_id,
                assistant_message_id=assistant_message_id,
                service=service,
                llm_profile=llm_profile,
                memory_llm_profile=memory_llm_profile,
                paper_search_source_limits=paper_search_source_limits,
            )
        except Exception:
            logger.exception(
                "Detached Agent task crashed: conversation_id=%s, run_id=%s",
                conversation_id,
                run_id,
            )
        finally:
            stream_run.finish()
            self._unregister_stream_run(stream_run)
            self._finish_conversation_run(conversation_id)

    async def _run_stream_worker(
        self,
        *,
        stream_run: DetachedStreamRun,
        message: str | None,
        conversation_id: str,
        run_id: str,
        resume_payload: dict[str, Any] | None,
        resume_action_id: str | None,
        assistant_message_id: int,
        service: AgentService,
        llm_profile: dict[str, Any] | None,
        memory_llm_profile: dict[str, Any] | None,
        paper_search_source_limits: dict[str, int] | None,
    ) -> None:
        started_at = time.perf_counter()
        db: DbSession | None = None
        try:
            db = self.db_factory()
            session = (
                Session.resume(
                    conversation_id=conversation_id,
                    run_id=run_id,
                    resume_payload=resume_payload,
                    db=db,
                    assistant_message_id=assistant_message_id,
                    llm_profile=llm_profile,
                    memory_llm_profile=memory_llm_profile,
                )
                if resume_payload is not None
                else Session.create(
                    message=message or "",
                    conversation_id=conversation_id,
                    run_id=run_id,
                    db=db,
                    user_id=DEFAULT_USER_ID,
                    assistant_message_id=assistant_message_id,
                    llm_profile=llm_profile,
                    memory_llm_profile=memory_llm_profile,
                    paper_search_source_limits=paper_search_source_limits,
                )
            )
            await self._consume_detached_stream(
                stream_run=stream_run,
                session=session,
                assistant_message_id=assistant_message_id,
                started_at=started_at,
                extra_meta=(
                    self._resume_meta(resume_payload, resume_action_id)
                    if resume_payload is not None
                    else self._llm_profile_meta(session)
                ),
                service=service,
            )
        except Exception as exc:
            logger.exception(
                "Detached Agent task failed: conversation_id=%s, run_id=%s",
                conversation_id,
                run_id,
            )
            response = await self._persist_detached_failure(
                assistant_message_id=assistant_message_id,
                conversation_id=conversation_id,
                run_id=run_id,
                latency_ms=self._elapsed_ms(started_at),
                error=str(exc),
                extra_meta=(
                    self._resume_meta(resume_payload, resume_action_id)
                    if resume_payload is not None
                    else None
                ),
            )
            stream_run.publish(
                Session(
                    conversation_id=conversation_id,
                    run_id=run_id,
                    assistant_message_id=assistant_message_id,
                    db=db,
                ).encode_sse("run_failed", response)
            )
        finally:
            if db is not None:
                db.close()

    async def _consume_detached_stream(
        self,
        *,
        stream_run: DetachedStreamRun,
        session: Session,
        assistant_message_id: int,
        started_at: float,
        extra_meta: dict[str, Any] | None,
        service: AgentService,
    ) -> None:
        terminal_persisted = False

        async def persist_terminal(
            response: dict[str, Any],
            card_meta: dict[str, Any],
        ) -> None:
            nonlocal terminal_persisted
            if terminal_persisted:
                return

            await self.history_store.complete_assistant_message(
                message_id=assistant_message_id,
                response=response,
                latency_ms=self._elapsed_ms(started_at),
                extra_meta={**card_meta, **(extra_meta or {})},
            )
            terminal_persisted = True
            self._schedule_consolidation(session=session, response=response)

        try:
            await self._stream_with_service(
                service=service,
                stream_run=stream_run,
                session=session,
                persist_terminal=persist_terminal,
            )
        except Exception as exc:
            logger.exception(
                "Detached Agent stream failed: conversation_id=%s, run_id=%s",
                session.conversation_id,
                session.run_id,
            )
            if terminal_persisted:
                return

            response = await self._persist_detached_failure(
                assistant_message_id=assistant_message_id,
                conversation_id=session.conversation_id,
                run_id=str(session.run_id),
                latency_ms=self._elapsed_ms(started_at),
                error=str(exc),
                extra_meta=extra_meta,
            )
            stream_run.publish(session.encode_sse("run_failed", response))

    async def _stream_with_service(
        self,
        *,
        service: AgentService,
        stream_run: DetachedStreamRun,
        session: Session,
        persist_terminal,
    ) -> None:
        async for event in service.stream(
            session,
            on_terminal=persist_terminal,
        ):
            stream_run.publish(event)

    async def _persist_detached_failure(
        self,
        *,
        assistant_message_id: int,
        conversation_id: str,
        run_id: str,
        latency_ms: int,
        error: str,
        extra_meta: dict[str, Any] | None,
    ) -> dict[str, Any]:
        response = {
            "conversation_id": conversation_id,
            "run_id": run_id,
            "status": "failed",
            "reply": "本次任务执行失败，请稍后重试。",
            "pending_action": None,
            "last_action_result": None,
            "artifact_refs": [],
            "interrupt": None,
            "error": error,
        }
        await self.history_store.complete_assistant_message(
            message_id=assistant_message_id,
            response=response,
            latency_ms=latency_ms,
            extra_meta={
                "schema_version": 1,
                "card": {
                    "status": "failed",
                    "reasoning": [],
                    "tools": [],
                    "subagents": [],
                    "memory": None,
                    "error": error,
                },
                **(extra_meta or {}),
            },
        )
        return response

    async def _resume_session(
        self,
        *,
        conversation_id: str,
        resume_payload: dict[str, Any],
        requested_run_id: str | None,
        requested_action_id: str | None,
        db: DbSession,
    ) -> tuple[Session, str, AgentService]:
        lookup = Session.for_resume_lookup(
            conversation_id=conversation_id,
            db=db,
        )
        snapshot = await self.agent_service.get_state(lookup)

        if not snapshot.values or not snapshot.next:
            raise ValueError("当前会话没有等待恢复的运行")

        run_id = str(snapshot.values["run_id"])
        if requested_run_id is not None and requested_run_id != run_id:
            raise ValueError("The requested run does not match the pending run")

        active_tool_call = snapshot.values.get("active_tool_call")
        if not isinstance(active_tool_call, dict):
            raise ValueError("The current run has no pending action")
        action_id = str(active_tool_call.get("id") or "")
        if not action_id:
            raise ValueError("The current pending action has no id")
        if requested_action_id is not None and requested_action_id != action_id:
            raise ValueError("The requested action does not match the pending action")

        profile_snapshot = snapshot.values.get("llm_profile")
        profile_id = (
            profile_snapshot.get("profile_id")
            if isinstance(profile_snapshot, dict)
            else None
        )
        llm_config = resolve_runtime_config(db, profile_id)
        memory_profile_snapshot = snapshot.values.get("memory_llm_profile")
        if isinstance(memory_profile_snapshot, dict):
            memory_profile_id = memory_profile_snapshot.get("profile_id")
            memory_llm_config = resolve_runtime_config(db, memory_profile_id)
        else:
            memory_llm_config = resolve_small_model_runtime_config(db)
            memory_profile_snapshot = self._memory_llm_profile_snapshot(
                llm_config,
                memory_llm_config,
            )
        return (
            Session.resume(
                conversation_id=conversation_id,
                run_id=run_id,
                resume_payload=resume_payload,
                db=db,
                llm_profile=(llm_config.snapshot() if llm_config else profile_snapshot),
                memory_llm_profile=(
                    memory_llm_config.snapshot()
                    if memory_llm_config
                    else memory_profile_snapshot
                ),
            ),
            action_id,
            self._service_for_config(llm_config, memory_llm_config),
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
        action_id: str | None = None,
    ) -> dict[str, Any]:
        resume = {
            "decision": resume_payload.get("decision"),
            "comment": resume_payload.get("comment"),
            "action_id": action_id,
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

    def _service_for_config(
        self,
        llm_config: LlmRuntimeConfig | None,
        memory_llm_config: LlmRuntimeConfig | None,
    ) -> AgentService:
        if llm_config is None and memory_llm_config is None:
            return self.agent_service
        return AgentService(
            checkpointer=self.agent_service.checkpointer,
            llm_config=llm_config,
            memory_llm_config=memory_llm_config,
        )

    @staticmethod
    def _llm_profile_meta(session: Session) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if session.llm_profile:
            metadata["llm_profile"] = session.llm_profile
        if session.memory_llm_profile:
            metadata["memory_llm_profile"] = session.memory_llm_profile
        return metadata

    @staticmethod
    def _memory_llm_profile_snapshot(
        llm_config: LlmRuntimeConfig | None,
        memory_llm_config: LlmRuntimeConfig | None,
    ) -> dict[str, Any] | None:
        effective_config = memory_llm_config or llm_config
        return effective_config.snapshot() if effective_config else None


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
