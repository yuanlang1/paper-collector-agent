from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from fastapi import FastAPI
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.api import agent as agent_api
from app.events.adapter import now
from app.events.bus import EventBus
from app.events.context import EventContext, bind_event_context, current_event_context
from app.events.models import (
    ActionResultEvent,
    ActionStartedEvent,
    ContentDeltaEvent,
    RunStartedEvent,
    RunTerminalEvent,
)
from app.history import chat_log, stream_events, transactions
from app.history.chat_log import AgentHistoryStore
from app.history.sqlite import HistoryDatabase
from app.llm.agent import AgentService
from app.llm.graph.main.nodes.subgraph import SubAgentNode
from app.llm.graph.main.nodes.solve import SolveNode
from app.llm.subagents.registry import (
    SubAgentRegistry,
    SubAgentRuntime,
    SubAgentSpec,
    SubAgentStreamSpec,
)
from app.llm.tools.registry import ToolRegistry
from app.runtime.agent_runtime import AgentRuntime
from app.runtime.session import Session
from app.database import Base
from app.models.agent_history import AgentConversation, AgentMessage, AgentRun


async def _append(events: list, event) -> None:
    events.append(event)


def _seed_conversation(db, *, user_id: str, conversation_id: str, thread_id: str | None = None) -> None:
    _, user_number = chat_log.normalize_user_id(user_id)
    existing = db.scalar(select(AgentConversation).where(
        AgentConversation.creator_id == user_number,
        AgentConversation.conversation_id == conversation_id,
    ))
    if existing is None:
        db.add(AgentConversation(
            creator_id=user_number,
            modifier_id=user_number,
            conversation_id=conversation_id,
            thread_id=thread_id or f"thread-{conversation_id}",
            status="ACTIVE",
        ))


class _TrackingDb:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _StreamingService:
    def __init__(self, *, block: bool = False, status: str = "completed") -> None:
        self.block = block
        self.status = status
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = asyncio.Event()
        self.sessions: list[Session] = []

    async def stream(self, session: Session, *, event_bus: EventBus) -> dict:
        self.sessions.append(session)
        self.started.set()
        try:
            await event_bus.publish(RunStartedEvent(
                run_id=str(session.run_id),
                ts=now(),
                payload={"status": "running"},
            ))
            if self.block:
                await self.release.wait()

            pending_action = None
            if self.status == "confirmation_required":
                pending_action = {
                    "action_id": "call-confirm-test",
                    "action_type": "tool",
                    "name": "test_tool",
                }
                await event_bus.publish(ActionStartedEvent(
                    run_id=str(session.run_id),
                    ts=now(),
                    payload={
                        **pending_action,
                        "input": {},
                        "requires_confirmation": True,
                    },
                ))

            response = {
                "conversation_id": session.conversation_id,
                "run_id": session.run_id,
                "status": self.status,
                "reply": "done",
                "pending_action": pending_action,
                "last_action_result": None,
                "artifact_refs": [],
                "interrupt": None,
                "error": None,
            }
            await event_bus.publish(RunTerminalEvent(
                type=(
                    "run.waiting_for_confirmation"
                    if self.status == "confirmation_required"
                    else "run.finished"
                ),
                run_id=str(session.run_id),
                ts=now(),
                status=self.status,
                payload={
                    key: value
                    for key, value in response.items()
                    if key not in {"conversation_id", "run_id"}
                },
            ))
            return response
        finally:
            self.closed.set()


class _ToolStreamingService:
    async def stream(self, session: Session, *, event_bus: EventBus) -> dict:
        run_id = str(session.run_id)
        action = {
            "action_id": "toolu_01",
            "action_type": "tool",
            "name": "read_file",
            "input": {"path": "README.md"},
            "requires_confirmation": False,
        }
        result = {
            "action_id": "toolu_01",
            "action_type": "tool",
            "name": "read_file",
            "status": "success",
            "summary": "tool completed",
            "data": {"content": "# KamaClaude"},
            "artifact_refs": [],
            "retryable": False,
            "error_code": None,
            "error_message": None,
        }
        await event_bus.publish(RunStartedEvent(
            run_id=run_id,
            ts=now(),
            payload={"status": "running"},
        ))
        await event_bus.publish(ContentDeltaEvent(
            run_id=run_id,
            ts=now(),
            payload={"delta": "我先读取 README。"},
        ))
        await event_bus.publish(ActionStartedEvent(
            run_id=run_id,
            ts=now(),
            payload=action,
        ))
        await event_bus.publish(ActionResultEvent(
            run_id=run_id,
            ts=now(),
            payload=result,
        ))
        await event_bus.publish(ContentDeltaEvent(
            run_id=run_id,
            ts=now(),
            payload={"delta": "README 的主要内容是……"},
        ))
        response = {
            "conversation_id": session.conversation_id,
            "run_id": run_id,
            "status": "completed",
            "reply": "README 的主要内容是……",
            "pending_action": None,
            "artifact_refs": [],
            "error": None,
        }
        await event_bus.publish(RunTerminalEvent(
            type="run.finished",
            run_id=run_id,
            ts=now(),
            status="completed",
            payload={
                key: value
                for key, value in response.items()
                if key not in {"conversation_id", "run_id"}
            },
        ))
        return response


class _CompletedGraph:
    async def astream(self, *_args, **_kwargs):
        if False:
            yield None

    async def aget_state(self, *_args, **_kwargs):
        return SimpleNamespace(values={})


class _InterruptedGraph:
    async def astream(self, *_args, **_kwargs):
        yield (), "updates", {
            "dispatch": {
                "active_tool_call": {
                    "id": "call-confirm-1",
                    "name": "paper_search_agent",
                    "kind": "subagent",
                    "args": {"prompt": "RAG 论文"},
                    "requires_confirmation": True,
                },
            },
        }
        yield (), "updates", {
            "__interrupt__": [{
                "action_id": "call-confirm-1",
                "status": "pending",
            }],
        }


class _SubagentGraph:
    async def astream(self, *_args, **_kwargs):
        if False:
            yield None

    async def aget_state(self, config):
        return SimpleNamespace(
            values={
                "run_id": config["configurable"]["run_id"],
                "run_status": "completed",
                "reply": "done",
            }
        )


class _BlockingSubagentGraph:
    def __init__(self) -> None:
        self.progress_published = asyncio.Event()
        self.release = asyncio.Event()
        self.state = None

    async def ainvoke(self, _state, *, config):
        self.state = dict(_state)
        event_context = current_event_context().scoped(node="test_progress")
        await event_context.bus.publish(
            event_context.event(
                "subagent_progress",
                {
                    "phase": "search",
                    "phase_label": "检索中",
                    "status": "running",
                    "message": "检索中",
                    "iteration": None,
                    "data": {"phase_state": "started"},
                },
            ),
        )
        self.progress_published.set()
        await self.release.wait()
        return {
            "last_action_result": {
                "status": "success",
                "summary": "检索完成",
                "data": {"count": 1},
            },
        }


class _NeverEndingSubagentGraph:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def ainvoke(self, _state, *, config):
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _CapturingSubagentGraph:
    def __init__(self) -> None:
        self.states: list[dict] = []

    async def ainvoke(self, state, *, config):
        self.states.append(dict(state))
        return {
            "last_action_result": {
                "status": "success",
                "summary": "完成",
                "data": {},
            },
        }


class _SolveModel:
    async def astream(self, _messages):
        yield AIMessage(
            content="answer",
            additional_kwargs={"reasoning_content": "reasoning"},
        )


class AgentStreamLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.history_db = HistoryDatabase(Path(self.temp_dir.name) / "history.db")
        self.history_db.initialize()
        self.history_engine = create_engine(
            f"sqlite:///{Path(self.temp_dir.name) / 'agent-history.db'}"
        )
        Base.metadata.create_all(
            self.history_engine,
            tables=[
                AgentConversation.__table__,
                AgentRun.__table__,
                AgentMessage.__table__,
            ],
        )
        self.history_sessions = sessionmaker(bind=self.history_engine)
        self.history_store = AgentHistoryStore(self.history_sessions)
        self.db_sessions: list[_TrackingDb] = []
        self._run_number = 0

    def tearDown(self) -> None:
        self.history_engine.dispose()
        self.temp_dir.cleanup()

    def _runtime(self, service: _StreamingService) -> AgentRuntime:
        def create_db() -> _TrackingDb:
            db = _TrackingDb()
            self.db_sessions.append(db)
            return db

        runtime = AgentRuntime(
            agent_service=service,
            event_db=self.history_db,
            history_store=self.history_store,
            db_factory=create_db,
        )
        return runtime

    async def _start_chat(
        self,
        runtime: AgentRuntime,
        *,
        conversation_id: str,
        user_id: str = "0",
        run_id: str | None = None,
        message: str = "hello",
    ):
        self._run_number += 1
        run_id = run_id or f"run_{conversation_id}_{self._run_number}"
        await self.history_store.run(
            _seed_conversation,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        with (
            patch(
                "app.runtime.agent_runtime.resolve_runtime_config",
                return_value=None,
            ),
            patch(
                "app.runtime.agent_runtime.resolve_small_model_runtime_config",
                return_value=None,
            ),
            patch("app.runtime.agent_runtime.get_source_limits", return_value={}),
        ):
            return await runtime.chat_stream(
                message=message,
                user_id=user_id,
                conversation_id=conversation_id,
                run_id=run_id,
                db=Mock(),
            )

    async def _collect(self, events):
        return [event async for event in events]

    async def _assistant_status(self, conversation_id: str) -> str:
        messages, _ = await self.history_store.run(
            chat_log.list_messages,
            user_id="0",
            conversation_id=conversation_id,
            limit=10,
            before_id=None,
        )
        return next(message["status"] for message in messages if message["role"] == "assistant")

    def test_background_task_runs_without_subscriber(self) -> None:
        async def scenario() -> None:
            service = _StreamingService()
            runtime = self._runtime(service)
            subscription = await self._start_chat(runtime, conversation_id="conv-background")

            await asyncio.wait_for(service.closed.wait(), timeout=1)
            events = await self.history_db.run(
                stream_events.list_after,
                user_id="0",
                run_id=subscription.run_id,
                after_sequence=None,
                limit=100,
            )

            self.assertEqual([event.event_name for event in events], ["run_started", "run_completed"])
            self.assertEqual([event.sequence for event in events], [1, 2])
            self.assertEqual(await self._assistant_status("conv-background"), "completed")
            self.assertTrue(self.db_sessions[0].closed)
            await subscription.events.aclose()

        asyncio.run(scenario())

    def test_closing_subscription_does_not_cancel_agent(self) -> None:
        async def scenario() -> None:
            service = _StreamingService(block=True)
            runtime = self._runtime(service)
            subscription = await self._start_chat(runtime, conversation_id="conv-disconnect")

            await asyncio.wait_for(service.started.wait(), timeout=1)
            first_event = await anext(subscription.events)
            await subscription.events.aclose()
            self.assertEqual(first_event.event_name, "run_started")
            self.assertFalse(service.closed.is_set())

            service.release.set()
            await asyncio.wait_for(service.closed.wait(), timeout=1)
            self.assertEqual(await self._assistant_status("conv-disconnect"), "completed")
            await runtime.shutdown()

        asyncio.run(scenario())

    def test_shutdown_marks_running_message_interrupted(self) -> None:
        async def scenario() -> None:
            service = _StreamingService(block=True)
            runtime = self._runtime(service)
            subscription = await self._start_chat(runtime, conversation_id="conv-shutdown")

            await asyncio.wait_for(service.started.wait(), timeout=1)
            await runtime.shutdown()

            self.assertTrue(service.closed.is_set())
            self.assertEqual(await self._assistant_status("conv-shutdown"), "failed")
            self.assertTrue(self.db_sessions[0].closed)
            await subscription.events.aclose()

        asyncio.run(scenario())

    def test_reconnect_replays_only_events_after_cursor(self) -> None:
        async def scenario() -> None:
            service = _StreamingService()
            runtime = self._runtime(service)
            started = await self._start_chat(runtime, conversation_id="conv-replay")
            await asyncio.wait_for(service.closed.wait(), timeout=1)

            all_events = await self._collect(started.events)
            replay = await runtime.subscribe_stream(
                user_id="0",
                conversation_id="conv-replay",
                run_id=started.run_id,
                last_event_sequence=all_events[0].sequence,
            )
            missing_events = await self._collect(replay.events)

            self.assertEqual([event.event_name for event in missing_events], ["run_completed"])
            self.assertEqual(missing_events[0].data["run_id"], started.run_id)
            self.assertNotIn("conversation_id", missing_events[0].data)
            await runtime.shutdown()

        asyncio.run(scenario())

    def test_terminal_event_is_not_lost_after_a_partial_batch(self) -> None:
        async def scenario() -> None:
            service = _StreamingService(block=True)
            runtime = self._runtime(service)
            subscription = await self._start_chat(runtime, conversation_id="conv-terminal-race")

            await asyncio.wait_for(service.started.wait(), timeout=1)
            first = await anext(subscription.events)
            service.release.set()
            await asyncio.wait_for(service.closed.wait(), timeout=1)
            terminal = await anext(subscription.events)

            self.assertEqual(first.event_name, "run_started")
            self.assertEqual(terminal.event_name, "run_completed")
            await subscription.events.aclose()
            await runtime.shutdown()

        asyncio.run(scenario())

    def test_terminal_message_commits_before_independent_event_write(self) -> None:
        async def scenario() -> None:
            await self.history_store.run(
                _seed_conversation,
                user_id="0",
                conversation_id="conv-atomic",
            )
            await self.history_store.run(
                chat_log.start_turn,
                user_id="0",
                conversation_id="conv-atomic",
                run_id="run_atomic",
                user_content="hello",
            )
            response = {
                "status": "completed",
                "reply": "done",
                "artifact_refs": [],
                "pending_action": None,
                "last_action_result": None,
                "error": None,
            }
            await self.history_store.run(
                transactions.complete_assistant_message,
                user_id="0",
                conversation_id="conv-atomic",
                run_id="run_atomic",
                response=response,
                latency_ms=1,
                extra_meta=None,
            )

            messages, _ = await self.history_store.run(
                chat_log.list_messages,
                user_id="0",
                conversation_id="conv-atomic",
                limit=10,
                before_id=None,
            )
            self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
            self.assertEqual(
                await self.history_db.run(
                    stream_events.list_after,
                    user_id="0",
                    run_id="run_atomic",
                    after_sequence=None,
                    limit=100,
                ),
                [],
            )

        asyncio.run(scenario())

    def test_user_isolation_covers_events_and_cursor_validation(self) -> None:
        async def scenario() -> None:
            await self.history_store.run(
                _seed_conversation,
                user_id="1",
                conversation_id="same-conversation",
            )
            await self.history_store.run(
                chat_log.start_turn,
                user_id="1",
                conversation_id="same-conversation",
                run_id="run-a",
                user_content="a",
            )
            event = await self.history_db.run(
                stream_events.append,
                user_id="1",
                conversation_id="same-conversation",
                run_id="run-a",
                event_name="run_started",
                data={"status": "running"},
            )
            runtime = self._runtime(_StreamingService())

            with self.assertRaises(LookupError):
                await runtime.subscribe_stream(
                    user_id="2",
                    conversation_id="same-conversation",
                    run_id="run-a",
                    last_event_sequence=None,
                )
            with self.assertRaises(ValueError):
                await runtime.subscribe_stream(
                    user_id="1",
                    conversation_id="same-conversation",
                    run_id="run-a",
                    last_event_sequence=event.sequence + 1,
                )
            self.assertEqual(
                await self.history_db.run(
                    stream_events.list_after,
                    user_id="2",
                    run_id="run-a",
                    after_sequence=None,
                    limit=100,
                ),
                [],
            )

        asyncio.run(scenario())

    def test_existing_events_are_backfilled_with_run_sequences(self) -> None:
        database_path = Path(self.temp_dir.name) / "legacy-history.db"
        connection = sqlite3.connect(database_path)
        connection.executescript(
            """
            CREATE TABLE agent_stream_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                event_name TEXT NOT NULL,
                data TEXT NOT NULL
            );
            INSERT INTO agent_stream_events (
                user_id, conversation_id, run_id, event_name, data
            ) VALUES
                ('user-1', 'conv-1', 'run-1', 'run_started', '{}'),
                ('user-1', 'conv-2', 'run-2', 'run_started', '{}'),
                ('user-1', 'conv-1', 'run-1', 'run_completed', '{}');
            """
        )
        connection.close()

        legacy_db = HistoryDatabase(database_path)
        legacy_db.initialize()

        async def scenario() -> None:
            first_run = await legacy_db.run(
                stream_events.list_after,
                user_id="user-1",
                run_id="run-1",
                after_sequence=None,
                limit=100,
            )
            second_run = await legacy_db.run(
                stream_events.list_after,
                user_id="user-1",
                run_id="run-2",
                after_sequence=None,
                limit=100,
            )
            self.assertEqual([event.sequence for event in first_run], [1, 2])
            self.assertEqual([event.sequence for event in second_run], [1])

        asyncio.run(scenario())
        legacy_db.initialize()

    def test_concurrent_writes_allocate_distinct_run_sequences(self) -> None:
        async def scenario() -> None:
            await self.history_store.run(
                _seed_conversation,
                user_id="1",
                conversation_id="conv-concurrent",
            )
            await self.history_store.run(
                chat_log.start_turn,
                user_id="1",
                conversation_id="conv-concurrent",
                run_id="run-concurrent",
                user_content="hello",
            )
            events = await asyncio.gather(
                *(
                    self.history_db.run(
                        stream_events.append,
                        user_id="1",
                        conversation_id="conv-concurrent",
                        run_id="run-concurrent",
                        event_name="content_delta",
                        data={"delta": str(index)},
                    )
                    for index in range(2)
                )
            )
            self.assertEqual(sorted(event.sequence for event in events), [1, 2])

        asyncio.run(scenario())

    def test_concurrent_conversation_runs_are_rejected(self) -> None:
        async def scenario() -> None:
            service = _StreamingService(block=True)
            runtime = self._runtime(service)
            first = await self._start_chat(runtime, conversation_id="conv-lock")
            await asyncio.wait_for(service.started.wait(), timeout=1)
            with self.assertRaises(chat_log.ActiveRunConflictError):
                await self._start_chat(runtime, conversation_id="conv-lock")
            service.release.set()
            await asyncio.gather(*runtime._active_stream_tasks.values())

            messages, _ = await self.history_store.run(
                chat_log.list_messages,
                user_id="0",
                conversation_id="conv-lock",
                limit=10,
                before_id=None,
            )
            self.assertEqual(
                [message["status"] for message in messages if message["role"] == "assistant"],
                ["completed"],
            )
            first_events = await self.history_db.run(
                stream_events.list_after,
                user_id="0",
                run_id=first.run_id,
                after_sequence=None,
                limit=100,
            )
            self.assertEqual([event.sequence for event in first_events], [1, 2])
            await first.events.aclose()
            await runtime.shutdown()

        asyncio.run(scenario())

    def test_confirmation_terminal_event_is_persisted(self) -> None:
        async def scenario() -> None:
            service = _StreamingService(status="confirmation_required")
            runtime = self._runtime(service)
            subscription = await self._start_chat(runtime, conversation_id="conv-confirm")
            events = await self._collect(subscription.events)

            self.assertEqual(events[-1].event_name, "confirmation_required")
            run_state = await self.history_store.run(
                chat_log.get_run_event_state,
                user_id="0",
                run_id=subscription.run_id,
            )
            self.assertIsNotNone(run_state)
            self.assertEqual(run_state.status, "confirmation_required")
            claim = await self.history_store.run(
                chat_log.claim_pending_action,
                user_id="0",
                conversation_id="conv-confirm",
                run_id=subscription.run_id,
                action_id="call-confirm-test",
                decision="approved",
                comment=None,
            )
            duplicate = await self.history_store.run(
                chat_log.claim_pending_action,
                user_id="0",
                conversation_id="conv-confirm",
                run_id=subscription.run_id,
                action_id="call-confirm-test",
                decision="approved",
                comment=None,
            )
            self.assertTrue(claim)
            self.assertFalse(duplicate)
            await runtime.shutdown()

        asyncio.run(scenario())

    def test_chat_log_records_text_tools_and_results_in_order(self) -> None:
        async def scenario() -> None:
            runtime = self._runtime(_ToolStreamingService())
            subscription = await self._start_chat(
                runtime,
                conversation_id="conv-tool-history",
                message="读取 README.md",
            )
            await self._collect(subscription.events)
            messages, _ = await self.history_store.run(
                chat_log.list_messages,
                user_id="0",
                conversation_id="conv-tool-history",
                limit=20,
                before_id=None,
            )

            self.assertEqual(
                [(message["role"], message["status"]) for message in messages],
                [
                    ("user", "completed"),
                    ("assistant", "completed"),
                    ("assistant", "completed"),
                    ("user", "completed"),
                    ("assistant", "completed"),
                ],
            )
            self.assertEqual(messages[0]["content"], [{"type": "text", "text": "读取 README.md"}])
            self.assertEqual(messages[1]["content"], [{"type": "text", "text": "我先读取 README。"}])
            self.assertEqual(messages[2]["content"][0]["type"], "tool_use")
            self.assertEqual(messages[2]["content"][0]["id"], "toolu_01")
            self.assertEqual(messages[3]["content"], [{
                "type": "tool_result",
                "tool_use_id": "toolu_01",
                "content": "# KamaClaude",
                "is_error": False,
            }])
            self.assertEqual(messages[4]["content"], [{
                "type": "text",
                "text": "README 的主要内容是……",
            }])
            self.assertTrue(all(
                not message["meta"] or "card" not in message["meta"]
                for message in messages
            ))
            turns = await self.history_store.run(
                chat_log.list_completed_turns_after,
                user_id="0",
                conversation_id="conv-tool-history",
                after_assistant_message_id=None,
                limit=10,
            )
            self.assertEqual(len(turns), 1)
            self.assertEqual(turns[0].assistant_content, "README 的主要内容是……")
            await runtime.shutdown()

        asyncio.run(scenario())

    def test_chat_log_preserves_subagent_result_details(self) -> None:
        async def scenario() -> None:
            await self.history_store.run(
                _seed_conversation,
                user_id="0",
                conversation_id="conv-subagent-history",
            )
            await self.history_store.run(
                chat_log.start_turn,
                user_id="0",
                conversation_id="conv-subagent-history",
                run_id="run-subagent-history",
                user_content="generate review",
            )
            action = {
                "action_id": "delegation-1",
                "action_type": "subagent",
                "name": "task_review_agent",
                "input": {},
                "requires_confirmation": False,
            }
            await self.history_store.run(
                transactions.persist_message_event,
                user_id="0",
                conversation_id="conv-subagent-history",
                run_id="run-subagent-history",
                event_name="action_started",
                event_data=action,
            )
            await self.history_store.run(
                transactions.persist_message_event,
                user_id="0",
                conversation_id="conv-subagent-history",
                run_id="run-subagent-history",
                event_name="action_result",
                event_data={
                    **action,
                    "action_type": "subagent",
                    "name": "task_review_agent",
                    "status": "error",
                    "summary": "文献综述工作流执行失败，未完成结果保存。",
                    "data": {"task_id": 12, "review_id": None},
                    "retryable": True,
                    "error_code": "WRITING_REVIEW_BLOCKED",
                    "error_message": "综述写作质量审核未通过。",
                },
            )

            messages, _ = await self.history_store.run(
                chat_log.list_messages,
                user_id="0",
                conversation_id="conv-subagent-history",
                limit=10,
                before_id=None,
            )

            self.assertEqual(messages[2]["content"], [{
                "type": "tool_result",
                "tool_use_id": "delegation-1",
                "content": {
                    "task_id": 12,
                    "review_id": None,
                    "summary": "文献综述工作流执行失败，未完成结果保存。",
                    "error_code": "WRITING_REVIEW_BLOCKED",
                    "error_message": "综述写作质量审核未通过。",
                },
                "is_error": True,
            }])

        asyncio.run(scenario())


class NativeSseResponseTests(unittest.TestCase):
    def test_agent_api_exposes_run_commands_and_event_subscription(self) -> None:
        app = FastAPI()
        app.include_router(agent_api.router)
        schema = app.openapi()
        paths = set(schema["paths"])

        self.assertIn("/api/agent/runs", paths)
        self.assertIn("/api/agent/runs/{run_id}/resume", paths)
        self.assertIn("/api/agent/runs/{run_id}/events", paths)
        self.assertIn("/api/agent/chat/stream", paths)
        self.assertIn("/api/agent/chat/resume/stream", paths)
        self.assertTrue(schema["paths"]["/api/agent/chat/stream"]["post"]["deprecated"])
        self.assertTrue(schema["paths"]["/api/agent/chat/resume/stream"]["post"]["deprecated"])
        self.assertNotIn("/api/agent/chat", paths)
        self.assertNotIn("/api/agent/chat/resume", paths)

    def test_agent_api_requires_request_context(self) -> None:
        app = FastAPI()
        app.include_router(agent_api.router)

        async def scenario() -> None:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/agent/chat/stream",
                    json={"message": "hello"},
                )
            self.assertEqual(response.status_code, 422)

        asyncio.run(scenario())

    def test_server_sent_event_uses_id_event_and_json_data(self) -> None:
        app = FastAPI()

        @app.get("/events", response_class=EventSourceResponse)
        async def events():
            yield ServerSentEvent(
                id="9223372036854775807",
                event="run_completed",
                data={"status": "completed"},
            )

        async def scenario() -> None:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/events")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
            self.assertIn("id: 9223372036854775807", response.text)
            self.assertIn("event: run_completed", response.text)
            data_line = next(
                line for line in response.text.splitlines() if line.startswith("data: ")
            )
            self.assertEqual(json.loads(data_line.removeprefix("data: ")), {"status": "completed"})

        asyncio.run(scenario())

    def test_agent_api_starts_run_and_replays_by_cursor(self) -> None:
        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as directory:
                history_db = HistoryDatabase(Path(directory) / "history.db")
                history_db.initialize()
                history_engine = create_engine(f"sqlite:///{Path(directory) / 'agent-history.db'}")
                Base.metadata.create_all(
                    history_engine,
                    tables=[
                        AgentConversation.__table__,
                        AgentRun.__table__,
                        AgentMessage.__table__,
                    ],
                )
                history_store = AgentHistoryStore(sessionmaker(bind=history_engine))
                await history_store.run(
                    _seed_conversation,
                    user_id="0",
                    conversation_id="conv-api",
                )
                service = _StreamingService()
                runtime = AgentRuntime(
                    agent_service=service,
                    event_db=history_db,
                    history_store=history_store,
                    db_factory=_TrackingDb,
                )
                app = FastAPI()
                app.include_router(agent_api.router)
                app.dependency_overrides[agent_api.get_db] = lambda: Mock()
                transport = httpx.ASGITransport(app=app)

                with (
                    patch.object(agent_api, "get_agent_runtime", return_value=runtime),
                    patch(
                        "app.runtime.agent_runtime.resolve_runtime_config",
                        return_value=None,
                    ),
                    patch(
                        "app.runtime.agent_runtime.resolve_small_model_runtime_config",
                        return_value=None,
                    ),
                    patch(
                        "app.runtime.agent_runtime.get_source_limits",
                        return_value={},
                    ),
                ):
                    async with httpx.AsyncClient(
                        transport=transport,
                        base_url="http://test",
                    ) as client:
                        response = await client.post(
                            "/api/agent/runs",
                            json={
                                "message": "hello",
                                "conversation_id": "conv-api",
                                "run_id": "run-api",
                            },
                        )
                        duplicate = await client.post(
                            "/api/agent/runs",
                            json={
                                "message": "hello",
                                "conversation_id": "conv-api",
                                "run_id": "run-api",
                            },
                        )
                        await asyncio.wait_for(service.closed.wait(), timeout=1)
                        run_id = response.json()["data"]["run_id"]
                        events = await history_db.run(
                            stream_events.list_after,
                            user_id="0",
                            run_id=run_id,
                            after_sequence=None,
                            limit=100,
                        )
                        replay = await client.get(
                            f"/api/agent/runs/{run_id}/events"
                            f"?conversation_id=conv-api&after_sequence={events[0].sequence}",
                        )

                self.assertEqual(response.status_code, 202)
                self.assertEqual(duplicate.status_code, 409)
                self.assertEqual(response.json()["data"]["conversation_id"], "conv-api")
                self.assertEqual(response.json()["data"]["status"], "started")
                self.assertIn("event: run_completed", replay.text)
                self.assertNotIn("event: run_started", replay.text)
                await runtime.shutdown()
                history_engine.dispose()

        asyncio.run(scenario())


class AgentServiceStreamTests(unittest.TestCase):
    def test_stream_exposes_terminal_persistence_data(self) -> None:
        async def scenario() -> None:
            service = AgentService.__new__(AgentService)
            service.subagent_registry = Mock()
            service.llm_config = None
            service.graph = _CompletedGraph()
            session = Session.create(
                message="hello",
                conversation_id="conv-terminal-error",
                run_id="run_terminal_error",
                db=Mock(),
            )

            events = []
            bus = EventBus()
            bus.subscribe(lambda event: _append(events, event))
            response = await service.stream(session, event_bus=bus)

            self.assertIsInstance(events[-1], RunTerminalEvent)
            self.assertEqual(response["status"], "completed")

        asyncio.run(scenario())

    def test_stream_publishes_confirmation_response(self) -> None:
        async def scenario() -> None:
            service = AgentService.__new__(AgentService)
            service.subagent_registry = None
            service.llm_config = None
            service.graph = _InterruptedGraph()
            session = Session.create(
                message="hello",
                conversation_id="conv-confirmation",
                run_id="run_confirmation",
                db=Mock(),
            )
            events = []
            bus = EventBus()
            bus.subscribe(lambda event: _append(events, event))

            response = await service.stream(session, event_bus=bus)

            event = events[-1]
            self.assertIsInstance(event, RunTerminalEvent)
            self.assertEqual(event.type, "run.waiting_for_confirmation")
            self.assertEqual(event.status, "confirmation_required")
            self.assertEqual(event.payload["pending_action"]["action_id"], "call-confirm-1")
            self.assertEqual(event.payload["interrupt"]["action_id"], "call-confirm-1")
            self.assertEqual(response["status"], "confirmation_required")

        asyncio.run(scenario())


class SubAgentNodeEventBusTests(unittest.TestCase):
    def test_all_subgraphs_receive_distinct_child_run_ids(self) -> None:
        async def scenario() -> None:
            workflows = ("paper_search", "task_indexing", "task_review")
            graphs = {workflow: _CapturingSubagentGraph() for workflow in workflows}
            registry = SubAgentRegistry(
                tuple(
                    SubAgentRuntime(
                        spec=SubAgentSpec(
                            name=f"{workflow}_agent",
                            description="test",
                            input_model=BaseModel,
                        ),
                        graph=graphs[workflow],
                        error_code="TEST_SUBAGENT_FAILED",
                        failure_summary="test failed",
                        stream=SubAgentStreamSpec(workflow=workflow),
                    )
                    for workflow in workflows
                )
            )
            node = SubAgentNode(subagent_registry=registry)
            with bind_event_context(EventContext(bus=EventBus(), run_id="run-1")):
                for workflow in workflows:
                    state = {
                        "run_id": "run-1",
                        "active_tool_call": {
                            "id": f"delegation-{workflow}",
                            "name": f"{workflow}_agent",
                            "kind": "subagent",
                        },
                    }
                    await node(state, {})
                    self.assertNotIn("child_run_id", state)

            child_run_ids = [
                graphs[workflow].states[0]["child_run_id"]
                for workflow in workflows
            ]
            self.assertEqual(len(set(child_run_ids)), len(workflows))
            self.assertTrue(
                all(
                    re.fullmatch(r"child_[0-9a-f]{16}_[0-9a-f]{32}", run_id)
                    for run_id in child_run_ids
                )
            )
            self.assertTrue(
                all(
                    graphs[workflow].states[0]["run_id"] == "run-1"
                    for workflow in workflows
                )
            )

        asyncio.run(scenario())

    def test_child_bus_publishes_progress_before_parent_continues(self) -> None:
        async def scenario() -> None:
            graph = _BlockingSubagentGraph()
            registry = SubAgentRegistry((
                SubAgentRuntime(
                    spec=SubAgentSpec(
                        name="paper_search_agent",
                        description="test",
                        input_model=BaseModel,
                    ),
                    graph=graph,
                    error_code="TEST_SUBAGENT_FAILED",
                    failure_summary="test failed",
                    stream=SubAgentStreamSpec(workflow="paper_search"),
                ),
            ))
            events = []
            parent_bus = EventBus()
            parent_bus.subscribe(lambda event: _append(events, event))
            node = SubAgentNode(subagent_registry=registry)
            state = {
                "run_id": "run-1",
                "active_tool_call": {
                    "id": "delegation-1",
                    "name": "paper_search_agent",
                    "kind": "subagent",
                },
            }

            with bind_event_context(
                EventContext(bus=parent_bus, run_id="run-1"),
            ):
                task = asyncio.create_task(node(state, {}))
                await asyncio.wait_for(graph.progress_published.wait(), timeout=1)
                self.assertFalse(task.done())
                self.assertEqual(
                    [event.payload.get("sse_name") for event in events],
                    ["subagent_started", "subagent_progress"],
                )
                self.assertTrue(all(
                    event.payload["delegation_id"] == "delegation-1"
                    for event in events
                ))
                started = events[0].payload
                self.assertEqual(started["subagent"], "paper_search_agent")
                self.assertEqual(started["parent_run_id"], "run-1")
                self.assertNotIn("name", started)
                self.assertNotIn("progress", started)
                self.assertEqual(started["progress_percent"], 0)
                self.assertEqual(graph.state["run_id"], "run-1")
                self.assertRegex(
                    graph.state["child_run_id"],
                    r"^child_[0-9a-f]{16}_[0-9a-f]{32}$",
                )
                self.assertEqual(
                    started["child_run_id"],
                    graph.state["child_run_id"],
                )

                graph.release.set()
                result = await task

            self.assertEqual(result["last_action_result"]["status"], "success")
            self.assertEqual(
                events[-1].payload["sse_name"],
                "subagent_completed",
            )
            completed = events[-1].payload
            self.assertNotIn("name", completed)
            self.assertNotIn("progress", completed)
            self.assertEqual(completed["progress_percent"], 100)

        asyncio.run(scenario())

    def test_timeout_cancels_child_and_returns_tool_result(self) -> None:
        async def scenario() -> None:
            graph = _NeverEndingSubagentGraph()
            registry = SubAgentRegistry((
                SubAgentRuntime(
                    spec=SubAgentSpec(
                        name="paper_search_agent",
                        description="test",
                        input_model=BaseModel,
                    ),
                    graph=graph,
                    error_code="TEST_SUBAGENT_FAILED",
                    failure_summary="test failed",
                    stream=SubAgentStreamSpec(workflow="paper_search"),
                    timeout_seconds=0.01,
                ),
            ))
            events = []
            parent_bus = EventBus()
            parent_bus.subscribe(lambda event: _append(events, event))
            node = SubAgentNode(subagent_registry=registry)
            state = {
                "run_id": "run-1",
                "active_tool_call": {
                    "id": "delegation-timeout",
                    "name": "paper_search_agent",
                    "kind": "subagent",
                    "args": {},
                },
            }

            with bind_event_context(EventContext(bus=parent_bus, run_id="run-1")):
                result = await node(state, {})

            await asyncio.wait_for(graph.cancelled.wait(), timeout=1)
            action_result = result["last_action_result"]
            self.assertEqual(action_result["status"], "error")
            self.assertTrue(action_result["retryable"])
            self.assertEqual(action_result["error_code"], "SUBAGENT_TIMEOUT")
            self.assertEqual(action_result["data"]["timeout_seconds"], 0.01)
            self.assertIsInstance(result["messages"][0], ToolMessage)
            self.assertEqual(result["messages"][0].tool_call_id, "delegation-timeout")
            self.assertEqual(
                [event.payload.get("sse_name") for event in events],
                ["subagent_started", "subagent_failed"],
            )

        asyncio.run(scenario())


class DirectEventPublicationTests(unittest.TestCase):
    def test_solve_publishes_events_to_bound_bus(self) -> None:
        async def scenario() -> None:
            events = []
            bus = EventBus()
            bus.subscribe(lambda event: _append(events, event))
            node = SolveNode(
                model=_SolveModel(),
                tool_registry=ToolRegistry(),
                subagent_registry=Mock(),
            )
            with bind_event_context(EventContext(bus=bus, run_id="run-1")):
                result = await node({
                    "run_id": "run-1",
                    "messages": [HumanMessage(content="hello")],
                    "system_context": "system",
                    "iteration_count": 0,
                })

            self.assertEqual(result["reply"], "answer")
            self.assertEqual(
                [event.type for event in events],
                ["iteration.started", "reasoning.delta", "content.delta"],
            )

        asyncio.run(scenario())

    def test_subagent_events_remain_in_stream_and_terminal_card(self) -> None:
        async def scenario() -> None:
            service = AgentService.__new__(AgentService)
            service.subagent_registry = Mock()
            service.llm_config = None
            service.graph = _SubagentGraph()
            session = Session.create(
                message="hello",
                conversation_id="conv-subagent-stream",
                run_id="run_subagent_stream",
                db=Mock(),
            )
            events = []
            bus = EventBus()
            bus.subscribe(lambda event: _append(events, event))
            await service.stream(session, event_bus=bus)

            self.assertEqual(
                [event.type for event in events],
                [
                    "run.started",
                    "run.finished",
                ],
            )

        asyncio.run(scenario())
