from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from fastapi import FastAPI
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.api import agent as agent_api
from app.history import chat_log, stream_events, transactions
from app.history.sqlite import HistoryDatabase
from app.llm.agent import AgentService
from app.runtime.agent_runtime import AgentRuntime
from app.runtime.session import Session


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

    async def stream(self, session: Session, *, on_terminal):
        self.sessions.append(session)
        self.started.set()
        try:
            yield {
                "event": "run_started",
                "status": "running",
                "conversation_id": session.conversation_id,
                "run_id": session.run_id,
            }
            if self.block:
                await self.release.wait()

            response = {
                "conversation_id": session.conversation_id,
                "run_id": session.run_id,
                "status": self.status,
                "reply": "done",
                "pending_action": None,
                "last_action_result": None,
                "artifact_refs": [],
                "interrupt": None,
                "error": None,
            }
            event_name = (
                "confirmation_required"
                if self.status == "confirmation_required"
                else "run_completed"
            )
            terminal = {"event": event_name, **response}
            await on_terminal(response, {"schema_version": 1, "card": {}}, terminal)
            yield terminal
        finally:
            self.closed.set()


class _CompletedGraph:
    async def astream(self, *_args, **_kwargs):
        if False:
            yield None

    async def aget_state(self, *_args, **_kwargs):
        return SimpleNamespace(values={})


class _SubagentGraph:
    async def astream(self, *_args, **_kwargs):
        yield (), "custom", {
            "event": "subagent_started",
            "delegation_id": "delegation-1",
            "workflow": "paper_search",
            "name": "论文检索",
        }
        yield (), "custom", {
            "event": "subagent_completed",
            "delegation_id": "delegation-1",
            "workflow": "paper_search",
            "summary": "检索完成",
        }

    async def aget_state(self, config):
        return SimpleNamespace(
            values={
                "run_id": config["configurable"]["run_id"],
                "run_status": "completed",
                "reply": "done",
            }
        )


class AgentStreamLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.history_db = HistoryDatabase(Path(self.temp_dir.name) / "history.db")
        self.history_db.initialize()
        self.db_sessions: list[_TrackingDb] = []

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _runtime(self, service: _StreamingService) -> AgentRuntime:
        def create_db() -> _TrackingDb:
            db = _TrackingDb()
            self.db_sessions.append(db)
            return db

        runtime = AgentRuntime(
            agent_service=service,
            history_db=self.history_db,
            db_factory=create_db,
        )
        runtime._schedule_consolidation = Mock()
        return runtime

    async def _start_chat(
        self,
        runtime: AgentRuntime,
        *,
        conversation_id: str,
        message: str = "hello",
    ):
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
                conversation_id=conversation_id,
                db=Mock(),
            )

    async def _collect(self, events):
        return [event async for event in events]

    async def _assistant_status(self, conversation_id: str) -> str:
        messages, _ = await self.history_db.run(
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
                after_id=None,
                limit=100,
            )

            self.assertEqual([event.event_name for event in events], ["run_started", "run_completed"])
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
            self.assertEqual(await self._assistant_status("conv-shutdown"), "interrupted")
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
                run_id=started.run_id,
                last_event_id=all_events[0].id,
            )
            missing_events = await self._collect(replay.events)

            self.assertEqual([event.event_name for event in missing_events], ["run_completed"])
            self.assertNotIn("run_id", missing_events[0].data)
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

    def test_terminal_update_and_event_insert_rollback_together(self) -> None:
        async def scenario() -> None:
            message_id = await self.history_db.run(
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
            with patch(
                "app.history.stream_events.insert",
                side_effect=RuntimeError("event write failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "event write failed"):
                    await self.history_db.run(
                        transactions.complete_assistant_message_with_event,
                        user_id="0",
                        conversation_id="conv-atomic",
                        run_id="run_atomic",
                        message_id=message_id,
                        response=response,
                        latency_ms=1,
                        extra_meta=None,
                        event_name="run_completed",
                        event_data={"status": "completed"},
                    )

            self.assertEqual(await self._assistant_status("conv-atomic"), "running")
            self.assertEqual(
                await self.history_db.run(
                    stream_events.list_after,
                    user_id="0",
                    run_id="run_atomic",
                    after_id=None,
                    limit=100,
                ),
                [],
            )

        asyncio.run(scenario())

    def test_user_isolation_covers_events_and_cursor_validation(self) -> None:
        async def scenario() -> None:
            await self.history_db.run(
                chat_log.start_turn,
                user_id="user-a",
                conversation_id="same-conversation",
                run_id="run-a",
                user_content="a",
            )
            event = await self.history_db.run(
                stream_events.append,
                user_id="user-a",
                conversation_id="same-conversation",
                run_id="run-a",
                event_name="run_started",
                data={"status": "running"},
            )
            runtime = self._runtime(_StreamingService())

            with self.assertRaises(LookupError):
                await runtime.subscribe_stream(
                    run_id="run-a",
                    last_event_id=None,
                    user_id="user-b",
                )
            with self.assertRaises(ValueError):
                await runtime.subscribe_stream(
                    run_id="run-a",
                    last_event_id=event.id + 1,
                    user_id="user-a",
                )
            self.assertEqual(
                await self.history_db.run(
                    stream_events.list_after,
                    user_id="user-b",
                    run_id="run-a",
                    after_id=None,
                    limit=100,
                ),
                [],
            )

        asyncio.run(scenario())

    def test_concurrent_conversation_runs_are_allowed(self) -> None:
        async def scenario() -> None:
            service = _StreamingService(block=True)
            runtime = self._runtime(service)
            first = await self._start_chat(runtime, conversation_id="conv-lock")
            await asyncio.wait_for(service.started.wait(), timeout=1)
            second = await self._start_chat(runtime, conversation_id="conv-lock")

            self.assertNotEqual(first.run_id, second.run_id)
            service.release.set()
            await asyncio.gather(*runtime._active_stream_tasks.values())

            messages, _ = await self.history_db.run(
                chat_log.list_messages,
                user_id="0",
                conversation_id="conv-lock",
                limit=10,
                before_id=None,
            )
            self.assertEqual(
                [message["status"] for message in messages if message["role"] == "assistant"],
                ["completed", "completed"],
            )
            await second.events.aclose()
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
            self.assertEqual(await self._assistant_status("conv-confirm"), "confirmation_required")
            await runtime.shutdown()

        asyncio.run(scenario())


class NativeSseResponseTests(unittest.TestCase):
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

    def test_agent_api_returns_run_header_and_replays_by_cursor(self) -> None:
        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as directory:
                history_db = HistoryDatabase(Path(directory) / "history.db")
                history_db.initialize()
                runtime = AgentRuntime(
                    agent_service=_StreamingService(),
                    history_db=history_db,
                    db_factory=_TrackingDb,
                )
                runtime._schedule_consolidation = Mock()
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
                            "/api/agent/chat/stream",
                            json={"message": "hello", "conversation_id": "conv-api"},
                        )
                        run_id = response.headers["x-agent-run-id"]
                        events = await history_db.run(
                            stream_events.list_after,
                            user_id="0",
                            run_id=run_id,
                            after_id=None,
                            limit=100,
                        )
                        replay = await client.get(
                            f"/api/agent/runs/{run_id}/events",
                            headers={"Last-Event-ID": str(events[0].id)},
                        )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["access-control-expose-headers"], "X-Agent-Run-ID")
                self.assertIn("event: run_started", response.text)
                self.assertIn("event: run_completed", response.text)
                self.assertIn("event: run_completed", replay.text)
                self.assertNotIn("event: run_started", replay.text)
                await runtime.shutdown()

        asyncio.run(scenario())


class AgentServicePersistenceTests(unittest.TestCase):
    def test_terminal_persistence_error_propagates(self) -> None:
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

            async def fail_terminal(*_args) -> None:
                raise RuntimeError("persistence failed")

            with self.assertRaisesRegex(RuntimeError, "persistence failed"):
                async for _event in service.stream(session, on_terminal=fail_terminal):
                    pass

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
            terminal_cards = []

            async def save_terminal(_response, card_meta, _event) -> None:
                terminal_cards.append(card_meta)

            events = [
                event
                async for event in service.stream(session, on_terminal=save_terminal)
            ]

            self.assertEqual(
                [event["event"] for event in events],
                [
                    "run_started",
                    "subagent_started",
                    "subagent_completed",
                    "run_completed",
                ],
            )
            self.assertEqual(
                terminal_cards[0]["card"]["subagents"][0]["workflow"],
                "paper_search",
            )

        asyncio.run(scenario())
