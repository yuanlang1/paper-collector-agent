from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from sqlalchemy import delete, select
from app.database import SessionLocal
from app.events.adapter import event_from_envelope
from app.events.bus import EventBus, RootEventDispatcher
from app.events.delivery import EventContext, TerminalUpdate
from app.events.handlers import IpcEventBroadcaster, PersistenceHandler
from app.events.models import RunTerminalEvent
from app.history import chat_log, stream_events, transactions
from app.history.sqlite import HistoryDatabase
from app.history.stream_events import AgentStreamEvent
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.provider import ChatClient
from app.memory.consolidation import Consolidator
from app.memory.extraction import LangChainMemoryExtractor
from app.models.memory import (
    MemoryConsolidationCursor,
    MemoryEpisode,
    MemoryFact,
)
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Callable, Optional
from sqlalchemy.orm import Session as DbSession
from app.llm.artifacts.access import ArtifactAccessService
from app.runtime.session import Session
from app.services.llm_profile_service import (
    LlmRuntimeConfig,
    resolve_runtime_config,
    resolve_small_model_runtime_config,
)
from app.services.setting_service import get_source_limits

if TYPE_CHECKING:
    from app.llm.agent import AgentService


logger = logging.getLogger(__name__)
DEFAULT_USER_ID = "0"


@dataclass(frozen=True)
class AgentStreamSubscription:
    run_id: str
    events: AsyncIterator[AgentStreamEvent]


class AgentRuntime:
    def __init__(
        self,
        agent_service: AgentService,
        history_db: HistoryDatabase,
        db_factory: Callable[[], DbSession] = SessionLocal,
        artifact_store: LocalArtifactStore | None = None,
    ) -> None:
        self.agent_service = agent_service
        self.history_db = history_db
        self.db_factory = db_factory
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.broadcaster = IpcEventBroadcaster(history_db)
        self._active_consolidation_scopes: set[tuple[str, str]] = set()
        self._active_consolidation_tasks: set[asyncio.Task[None]] = set()
        self._active_stream_tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        
    async def chat(
        self,
        *,
        message: str,
        user_id: str,
        conversation_id: str,
        run_id: str,
        db: DbSession,
        llm_profile_id: int | None = None,
    ) -> dict[str, Any]:
        llm_config = resolve_runtime_config(db, llm_profile_id)
        memory_llm_config = resolve_small_model_runtime_config(db)
        source_limits = get_source_limits(db)
        session = Session.create(
            message=message,
            conversation_id=conversation_id,
            run_id=run_id,
            db=db,
            user_id=user_id,
            llm_profile=llm_config.snapshot() if llm_config else None,
            memory_llm_profile=self._memory_llm_profile_snapshot(
                llm_config,
                memory_llm_config,
            ),
            paper_search_source_limits=source_limits,
        )

        assistant_message_id = await self.history_db.run(
            chat_log.start_turn,
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
            await self.history_db.run(
                chat_log.mark_interrupted,
                user_id=session.user_id,
                conversation_id=session.conversation_id,
                run_id=str(session.run_id),
                message_id=assistant_message_id,
            )
            raise

        await self.history_db.run(
            chat_log.complete_assistant_message,
            user_id=session.user_id,
            conversation_id=session.conversation_id,
            run_id=str(session.run_id),
            message_id=assistant_message_id,
            response=response,
            latency_ms=self._elapsed_ms(started_at),
            extra_meta=self._llm_profile_meta(session),
        )
        self._schedule_consolidation(session=session, response=response)

        return response

    async def chat_stream(
        self,
        *,
        message: str,
        user_id: str,
        conversation_id: str,
        run_id: str,
        db: DbSession,
        llm_profile_id: int | None = None,
    ) -> AgentStreamSubscription:
        llm_config = resolve_runtime_config(db, llm_profile_id)
        memory_llm_config = resolve_small_model_runtime_config(db)
        source_limits = get_source_limits(db)
        service = self._service_for_config(llm_config, memory_llm_config)
        llm_profile = llm_config.snapshot() if llm_config else None
        memory_llm_profile = self._memory_llm_profile_snapshot(
            llm_config,
            memory_llm_config,
        )
        db.rollback()
        assistant_message_id: int | None = None
        try:
            assistant_message_id = await self.history_db.run(
                chat_log.start_turn,
                user_id=user_id,
                conversation_id=conversation_id,
                run_id=run_id,
                user_content=message,
            )
            self._start_stream_task(
                message=message,
                user_id=user_id,
                conversation_id=conversation_id,
                run_id=run_id,
                resume_payload=None,
                assistant_message_id=assistant_message_id,
                service=service,
                llm_profile=llm_profile,
                memory_llm_profile=memory_llm_profile,
                paper_search_source_limits=source_limits,
                extra_meta=None,
            )
            return AgentStreamSubscription(
                run_id=run_id,
                events=self.broadcaster.iter_events(
                    user_id=user_id,
                    root_run_id=run_id,
                    after_sequence=None,
                ),
            )
        except BaseException:
            await self._abort_stream_setup(
                user_id=user_id,
                conversation_id=conversation_id,
                run_id=run_id,
                assistant_message_id=assistant_message_id,
            )
            raise

    async def resume_chat(
        self,
        *,
        user_id: str,
        conversation_id: str,
        resume_payload: dict[str, Any],
        requested_run_id: str | None = None,
        requested_action_id: str | None = None,
        db: DbSession,
    ) -> dict[str, Any]:
        session, action_id, service = await self._resume_session(
            user_id=user_id,
            conversation_id=conversation_id,
            resume_payload=resume_payload,
            requested_run_id=requested_run_id,
            requested_action_id=requested_action_id,
            db=db,
        )
        claim = await self.history_db.run(
            chat_log.claim_pending_action,
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
            await self.history_db.run(
                chat_log.mark_interrupted,
                user_id=session.user_id,
                conversation_id=session.conversation_id,
                run_id=str(session.run_id),
                message_id=assistant_message_id,
            )
            raise

        await self.history_db.run(
            chat_log.complete_assistant_message,
            user_id=session.user_id,
            conversation_id=session.conversation_id,
            run_id=str(session.run_id),
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

    async def resume_chat_stream(
        self,
        *,
        user_id: str,
        conversation_id: str,
        resume_payload: dict[str, Any],
        requested_run_id: str | None = None,
        requested_action_id: str | None = None,
        db: DbSession,
    ) -> AgentStreamSubscription:
        assistant_message_id: int | None = None
        run_id = requested_run_id or "resume-pending"
        try:
            session, action_id, service = await self._resume_session(
                user_id=user_id,
                conversation_id=conversation_id,
                resume_payload=resume_payload,
                requested_run_id=requested_run_id,
                requested_action_id=requested_action_id,
                db=db,
            )
            db.rollback()
            run_id = str(session.run_id)
            claim = await self.history_db.run(
                chat_log.claim_pending_action,
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
            self._start_stream_task(
                message=None,
                user_id=session.user_id,
                conversation_id=conversation_id,
                run_id=run_id,
                resume_payload=resume_payload,
                assistant_message_id=assistant_message_id,
                service=service,
                llm_profile=session.llm_profile,
                memory_llm_profile=session.memory_llm_profile,
                paper_search_source_limits=None,
                extra_meta=self._resume_meta(resume_payload, action_id),
            )
            return AgentStreamSubscription(
                run_id=run_id,
                events=self.broadcaster.iter_events(
                    user_id=session.user_id,
                    root_run_id=run_id,
                    after_sequence=None,
                ),
            )
        except BaseException:
            await self._abort_stream_setup(
                user_id=user_id,
                conversation_id=conversation_id,
                run_id=run_id,
                assistant_message_id=assistant_message_id,
            )
            raise

    async def delete_conversation(
        self,
        *,
        conversation_id: str,
        db: DbSession,
    ) -> dict[str, Any]:
        await self._run_conversation_deletion(
            user_id=DEFAULT_USER_ID,
            conversation_id=conversation_id,
            db=db,
        )

        return {
            "conversation_id": conversation_id,
            "deleted": True,
        }

    async def _run_conversation_deletion(
        self,
        *,
        user_id: str,
        conversation_id: str,
        db: DbSession,
    ) -> None:
        async def delete_artifacts() -> None:
            run_ids = await self.history_db.run(
                chat_log.list_run_ids,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            await self.artifact_store.delete_run_directories(run_ids)

        for resource, operation in (
            ("artifacts", delete_artifacts()),
            (
                "checkpoints",
                self.agent_service.checkpointer.adelete_thread(conversation_id),
            ),
            (
                "memory",
                self._delete_conversation_memory(
                    db=db,
                    conversation_id=conversation_id,
                ),
            ),
            (
                "history",
                self.history_db.run(
                    transactions.delete_conversation,
                    user_id=user_id,
                    conversation_id=conversation_id,
                ),
            ),
        ):
            try:
                await operation
            except Exception:
                logger.exception(
                    "Failed to delete conversation %s: conversation_id=%s",
                    resource,
                    conversation_id,
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
            if fact_ids or episode_ids:
                from app.rag.index_construction.memory_index_sync import (
                    get_memory_index_synchronizer,
                )

                synchronizer = get_memory_index_synchronizer()
                await synchronizer.delete_facts(fact_ids)
                await synchronizer.delete_episodes(episode_ids)
        except Exception:
            db.rollback()
            raise

    async def shutdown(self) -> None:
        tasks = list(self._active_stream_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

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
            result = await Consolidator(
                db,
                user_id=user_id,
                history_db=self.history_db,
                extraction_model=LangChainMemoryExtractor(
                    chat=ChatClient(memory_llm_config),
                ),
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

    async def _abort_stream_setup(
        self,
        *,
        user_id: str,
        conversation_id: str,
        run_id: str,
        assistant_message_id: int | None,
    ) -> None:
        try:
            if assistant_message_id is not None:
                await self.history_db.run(
                    chat_log.mark_interrupted,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    message_id=assistant_message_id,
                )
        except Exception:
            logger.exception(
                "Failed to interrupt prepared stream: user_id=%s, conversation_id=%s",
                user_id,
                conversation_id,
            )

    async def subscribe_stream(
        self,
        *,
        user_id: str,
        conversation_id: str,
        run_id: str,
        last_event_sequence: int | None,
    ) -> AgentStreamSubscription:
        scope = await self.history_db.run(chat_log.get_run_scope, run_id)
        if (
            scope is None
            or scope.user_id != user_id
            or scope.conversation_id != conversation_id
        ):
            raise LookupError("run not found")
        if last_event_sequence is not None:
            event = await self.history_db.run(
                stream_events.get,
                user_id=user_id,
                run_id=run_id,
                sequence=last_event_sequence,
            )
            if event is None:
                raise ValueError("Last-Event-ID does not belong to this run")
        return AgentStreamSubscription(
            run_id=run_id,
            events=self.broadcaster.iter_events(
                user_id=user_id,
                root_run_id=run_id,
                after_sequence=last_event_sequence,
            ),
        )

    def _start_stream_task(
        self,
        *,
        message: str | None,
        user_id: str,
        conversation_id: str,
        run_id: str,
        resume_payload: dict[str, Any] | None,
        assistant_message_id: int,
        service: AgentService,
        llm_profile: dict[str, Any] | None,
        memory_llm_profile: dict[str, Any] | None,
        paper_search_source_limits: dict[str, int] | None,
        extra_meta: dict[str, Any] | None,
    ) -> None:
        key = (user_id, run_id)
        task = asyncio.create_task(
            self._run_stream_task(
                message=message,
                user_id=user_id,
                conversation_id=conversation_id,
                run_id=run_id,
                resume_payload=resume_payload,
                assistant_message_id=assistant_message_id,
                service=service,
                llm_profile=llm_profile,
                memory_llm_profile=memory_llm_profile,
                paper_search_source_limits=paper_search_source_limits,
                extra_meta=extra_meta,
            ),
            name=f"agent-stream:{user_id}:{run_id}",
        )
        self._active_stream_tasks[key] = task

    async def _run_stream_task(
        self,
        *,
        message: str | None,
        user_id: str,
        conversation_id: str,
        run_id: str,
        resume_payload: dict[str, Any] | None,
        assistant_message_id: int,
        service: AgentService,
        llm_profile: dict[str, Any] | None,
        memory_llm_profile: dict[str, Any] | None,
        paper_search_source_limits: dict[str, int] | None,
        extra_meta: dict[str, Any] | None,
    ) -> None:
        started_at = time.perf_counter()
        terminal_persisted = False
        stream_session: Session | None = None
        db: DbSession | None = None
        context = EventContext(
            user_id=user_id,
            conversation_id=conversation_id,
            root_run_id=run_id,
            assistant_message_id=assistant_message_id,
        )
        bus = EventBus()
        bus.subscribe(
            RootEventDispatcher(
                context=context,
                persistence=PersistenceHandler(self.history_db, started_at=started_at),
                broadcaster=self.broadcaster,
            )
        )

        try:
            db = self.db_factory()
            stream_session = (
                Session.resume(
                    conversation_id=conversation_id,
                    run_id=run_id,
                    resume_payload=resume_payload,
                    db=db,
                    user_id=user_id,
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
                    user_id=user_id,
                    assistant_message_id=assistant_message_id,
                    llm_profile=llm_profile,
                    memory_llm_profile=memory_llm_profile,
                    paper_search_source_limits=paper_search_source_limits,
                )
            )
            async for event in service.stream(stream_session):
                if isinstance(event, RunTerminalEvent):
                    if event.terminal is None:
                        raise RuntimeError("Terminal event is missing persistence data")
                    event = event.model_copy(
                        update={
                            "terminal": TerminalUpdate(
                                response=event.terminal.response,
                                card_meta={
                                    **event.terminal.card_meta,
                                    **self._llm_profile_meta(stream_session),
                                    **(extra_meta or {}),
                                },
                            )
                        }
                    )
                await bus.publish(event)
                if isinstance(event, RunTerminalEvent):
                    terminal_persisted = True
                    self._schedule_consolidation(
                        session=stream_session,
                        response=event.terminal.response,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Agent stream failed: user_id=%s, conversation_id=%s, run_id=%s",
                user_id,
                conversation_id,
                run_id,
            )
            if not terminal_persisted:
                response = self._failed_stream_response(
                    conversation_id=conversation_id,
                    run_id=run_id,
                    error=str(exc),
                )
                event = event_from_envelope(
                    {"event": "run_failed", **response},
                    run_id=run_id,
                )
                if not isinstance(event, RunTerminalEvent):
                    raise RuntimeError("Failed to build a terminal event") from exc
                event = event.model_copy(
                    update={
                        "terminal": TerminalUpdate(
                            response=response,
                            card_meta={
                                "schema_version": 1,
                                "card": {},
                                **self._llm_profile_meta(stream_session),
                                **(extra_meta or {}),
                            },
                        )
                    }
                )
                try:
                    await bus.publish(event)
                    terminal_persisted = True
                except Exception:
                    logger.exception(
                        "Failed to persist stream failure: user_id=%s, run_id=%s",
                        user_id,
                        run_id,
                    )
        finally:
            try:
                if not terminal_persisted:
                    await asyncio.shield(
                        self.history_db.run(
                            chat_log.mark_interrupted,
                            user_id=user_id,
                            conversation_id=conversation_id,
                            run_id=run_id,
                            message_id=assistant_message_id,
                        )
                    )
            except Exception:
                logger.exception("Failed to clean up agent stream: run_id=%s", run_id)
            finally:
                if db is not None:
                    db.close()
                self._active_stream_tasks.pop((user_id, run_id), None)

    @staticmethod
    def _failed_stream_response(
        *,
        conversation_id: str,
        run_id: str,
        error: str,
    ) -> dict[str, Any]:
        return {
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

    async def _resume_session(
        self,
        *,
        user_id: str,
        conversation_id: str,
        resume_payload: dict[str, Any],
        requested_run_id: str | None,
        requested_action_id: str | None,
        db: DbSession,
    ) -> tuple[Session, str, AgentService]:
        lookup = Session.for_resume_lookup(
            conversation_id=conversation_id,
            db=db,
            user_id=user_id,
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
                user_id=user_id,
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
        from app.llm.agent import AgentService

        return AgentService(
            checkpointer=self.agent_service.checkpointer,
            llm_config=llm_config,
            memory_llm_config=memory_llm_config,
            artifact_access_service=self.agent_service.artifact_access_service,
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
    history_db: HistoryDatabase,
) -> None:
    global _agent_runtime

    from app.llm.agent import AgentService

    _agent_runtime = AgentRuntime(
        agent_service=AgentService(
            checkpointer=checkpointer,
            artifact_access_service=ArtifactAccessService(history_db),
        ),
        history_db=history_db,
    )

def get_agent_runtime() -> AgentRuntime:
    if _agent_runtime is None:
        raise RuntimeError("Agent runtime has not been initialized")
    return _agent_runtime
