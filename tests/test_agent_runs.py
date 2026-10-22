import asyncio
import json
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi import FastAPI
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt
from langsmith import tracing_context
from psycopg import AsyncConnection
from psycopg.conninfo import make_conninfo
from pydantic import ValidationError
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from starlette.requests import ClientDisconnect

from app.api.run import RunStreamingResponse, router as run_router
from app.api.thread import router as thread_router
from app.common.exceptions import ConflictException, register_exception_handlers
from app.database import get_db
from app.db.models import AgentRun, AgentThread, RunStatus
from app.db.repositories import agent_run, agent_thread
from app.llm import agent_build
from app.llm.agent_runner import AgentRunner
from app.llm.context import AgentRunContext
from app.llm.loop.agent_loop import AgentLoop
from app.llm.tools.registry import ToolRegistry
from app.schemas.ai import ChatRequest
from app.services import agent_run_service, llm_setting_service, thread_service
from app.services.llm_setting_service import LlmConfig
from tests.test_postgres_models import PostgresDatabaseTestCase, TEST_DATABASE_URL


def events(response):
    output = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    for event in output:
        assert "thread_id" not in event and "run_id" not in event
    return output


class RunInputTests(unittest.TestCase):
    def test_empty_thread_id_is_normalized_for_new_messages(self):
        fields = {"user_id": 1, "metadata": {"model_id": 7}, "input": {"message": "hi"}}
        for values in ({}, {"thread_id": None}, {"thread_id": ""}):
            with self.subTest(values=values):
                self.assertIsNone(ChatRequest(**fields, **values).thread_id)

    def test_resume_requires_thread_id_even_for_null_or_false_values(self):
        fields = {"user_id": 1, "metadata": {"model_id": 7}}
        for values in ({}, {"thread_id": None}, {"thread_id": ""}):
            for resume in (None, False, True):
                with self.subTest(values=values, resume=resume), self.assertRaises(ValidationError):
                    ChatRequest(**fields, **values, resume=resume)

    def test_exactly_one_input_or_resume_is_required(self):
        fields = {"user_id": 1, "metadata": {"model_id": 7}, "thread_id": "thread"}
        for input in (
            {}, {"input": None}, {"input": {}}, {"input": {"message": ""}},
            {"input": {"message": "hi"}, "resume": True}, {"user_id": 0},
            {"input": {"message": "hi"}, "metadata": {"model_id": 0}},
            {"message": "hi"},
        ):
            with self.subTest(input=input), self.assertRaises(ValidationError):
                ChatRequest(**(fields | input))
        self.assertEqual(ChatRequest(**fields, input={"message": "hi"}).input.message, "hi")
        self.assertIn("resume", ChatRequest(**fields, resume=None).model_fields_set)
        with self.assertRaises(ValidationError):
            ChatRequest(user_id=1, model_id=7, thread_id="thread", input={"message": "hi"})

    def test_metadata_requires_model_id_and_validates_effort(self):
        fields = {"user_id": 1, "thread_id": "thread", "input": {"message": "hi"}}
        for metadata in (None, {}, {"model_id": -1}, {"model_id": 7, "think_effort": "invalid"}):
            with self.subTest(metadata=metadata), self.assertRaises(ValidationError):
                ChatRequest(**fields, metadata=metadata)
        request = ChatRequest(**fields, metadata={"model_id": 7, "think_effort": "high"})
        self.assertEqual(request.metadata.think_effort, "high")
        resumed = ChatRequest(user_id=1, thread_id="thread", metadata={"model_id": 7}, resume=None)
        self.assertEqual(resumed.model_dump(include={"metadata", "resume"}, exclude_unset=True), {
            "metadata": {"model_id": 7}, "resume": None,
        })


class RunStreamInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_nested_input_and_hitl_resume_reach_the_graph(self):
        self.enterContext(tracing_context(enabled=False))
        hitl_request = {
            "action_requests": [{
                "name": "review_text", "args": {"content": "draft"},
                "description": "Review and edit the draft",
            }],
            "review_configs": [{
                "action_name": "review_text", "allowed_decisions": ["approve", "edit", "reject"],
            }],
        }
        answer = {"decisions": [{
            "type": "edit", "edited_action": {"name": "review_text", "args": {"content": "edited"}},
        }]}
        observed = []

        def work(state):
            observed.append(state["messages"][0].content)
            response = interrupt(hitl_request)
            return {"messages": [AIMessage(content=response["decisions"][0]["edited_action"]["args"]["content"])]}

        saver = InMemorySaver()
        builder = StateGraph(MessagesState)
        builder.add_node("work", work)
        builder.add_edge(START, "work")
        builder.add_edge("work", END)
        graph = builder.compile(checkpointer=saver)

        def build_loop(*, context, **kwargs):
            loop = object.__new__(AgentLoop)
            loop.context, loop.graph = context, graph
            return loop

        async def get_test_db():
            yield None

        app = FastAPI()
        app.include_router(run_router)
        app.dependency_overrides[get_db] = get_test_db
        app.state.session_factory = lambda: nullcontext(None)
        app.state.checkpointer, app.state.store = saver, None
        run = SimpleNamespace(user_id=1, thread_id="thread", run_id="run", run_metadata={"model_id": 7})
        fields = {"user_id": 1, "thread_id": "thread", "metadata": {"model_id": 7}}
        with patch.object(
            agent_run_service, "create_run", new=AsyncMock(return_value=run),
        ) as create_run, patch.object(
            agent_run_service, "update_status", new=AsyncMock(return_value=run),
        ), patch.object(
            agent_build, "get_model", new=AsyncMock(return_value=object()),
        ), patch.object(agent_build, "build_tools", return_value=ToolRegistry()), patch(
            "app.llm.agent_runner.AgentLoop", side_effect=build_loop,
        ):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                for invalid_input in (
                    {}, {"message": "review"}, {"input": {}},
                    {"input": {"message": "review"}, "resume": answer},
                ):
                    invalid = await client.post("/api/thread/runs/stream", json={**fields, **invalid_input})
                    self.assertEqual(invalid.status_code, 422)
                create_run.assert_not_awaited()
                paused = await client.post("/api/thread/runs/stream", json={**fields, "input": {"message": "review"}})
                self.assertEqual(paused.status_code, 200)
                paused_events = events(paused)
                self.assertEqual(paused_events[:2], [
                    {"type": "status", "status": "running"},
                    {"type": "status", "message": "正在思考…"},
                ])
                pending = paused_events[-1]["interrupts"][0]
                self.assertEqual(pending["value"], hitl_request)
                self.assertEqual(paused_events[-2], {**paused_events[-1], "type": "interrupt"})
                self.assertEqual(paused_events[-1]["status"], "interrupted")
                resumed = await client.post("/api/thread/runs/stream", json={
                    **fields, "resume": {pending["id"]: answer},
                })
        self.assertEqual(resumed.status_code, 200)
        self.assertEqual(events(resumed), [{"type": "done", "status": "completed", "interrupts": []}])
        snapshot = await graph.aget_state({"configurable": {"thread_id": "thread"}})
        self.assertEqual([message.content for message in snapshot.values["messages"]], ["review", "edited"])
        self.assertEqual(observed, ["review", "review"])


class AgentBuildTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_model_uses_owned_settings_and_constructs_chat_openai_directly(self):
        db = Mock()
        config = LlmConfig(provider="unused-provider", model="selected", base_url="https://example.test/v1", api_key="test-key")
        for effort in (None, "high"):
            with self.subTest(effort=effort), patch.object(
                agent_build.llm_setting_service, "get_llm_config", new_callable=AsyncMock, return_value=config,
            ) as get_config, patch.object(agent_build, "ChatOpenAI") as constructor:
                model = await agent_build.get_model(db, model_id=7, user_id=11, think_effort=effort)
                get_config.assert_awaited_once_with(db, 7, user_id=11)
                constructor.assert_called_once_with(
                    model="selected", base_url=config.base_url, api_key="test-key", streaming=True, reasoning_effort=effort,
                )
                self.assertIs(model, constructor.return_value)

    async def test_runner_uses_injected_model_and_tool_preview_names(self):
        model = FakeListChatModel(responses=["unused"])
        registry = agent_build.build_tools(Mock(), user_id=1, model=model)
        async def stream(**kwargs):
            yield {"type": "custom", "ns": (), "data": {
                "type": "tool_start", "name": "paper_search_agent", "tool_call_id": "call-1",
            }}
            yield {"type": "custom", "ns": (), "data": {
                "type": "tool_end", "name": "paper_search_agent", "preview": "found", "tool_call_id": "call-1",
            }}
        loop = SimpleNamespace(
            graph=SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(interrupts=()))),
            stream=stream,
        )
        context = AgentRunContext(thread_id="thread", run_id="run", user_id="1")
        runner = AgentRunner(context=context, model=model, tool_registry=registry, checkpointer=None, store=None)
        with patch("app.llm.agent_runner.AgentLoop", return_value=loop) as build_loop:
            output = [event async for event in runner.run({"message": "hi"})]
        self.assertIs(build_loop.call_args.kwargs["model"], model)
        self.assertIs(build_loop.call_args.kwargs["context"], context)
        self.assertEqual(list(build_loop.call_args.kwargs["tools"]), list(registry.get_tools()))
        self.assertEqual([event["label"] for event in output if event["type"].startswith("tool_")], ["论文检索子代理", "论文检索子代理"])
        self.assertEqual(output[-1]["preview"], "found")


class AgentRunTests(PostgresDatabaseTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.enterContext(tracing_context(enabled=False))
        # Raw checkpoint SQL and ORM statements both stay in the isolated schema.
        engine = create_async_engine(
            self.engine.url, connect_args={"options": f"-csearch_path={self.schema}"},
        )
        self.addAsyncCleanup(engine.dispose)
        self.factory = async_sessionmaker(engine, expire_on_commit=False)
        saver_context = AsyncPostgresSaver.from_conn_string(make_conninfo(
            TEST_DATABASE_URL, options=f"-csearch_path={self.schema}",
        ))
        self.saver = await saver_context.__aenter__()
        self.addAsyncCleanup(saver_context.__aexit__, None, None, None)
        await self.saver.setup()
        self.app = FastAPI()
        register_exception_handlers(self.app)
        self.app.state.session_factory = self.factory
        self.app.state.checkpointer = self.saver
        self.app.state.store = None
        self.app.include_router(run_router)
        self.app.include_router(thread_router)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")
        self.addAsyncCleanup(self.client.aclose)
        async with self.factory() as db:
            self.thread = await thread_service.create_thread(db, user_id=1, message="test")
            model = await llm_setting_service.create_setting(
                db, user_id=1, provider="unused-provider", base_url="https://example.test/v1",
                model="test-model", api_key="test-key",
            )
            self.model_id = model.id
        self.model_constructor = self.enterContext(patch("app.llm.agent_build.ChatOpenAI"))
        self.model_constructor.return_value = FakeListChatModel(responses=["hello"])
        self.tool_registry = ToolRegistry()
        self.tools_builder = self.enterContext(patch("app.llm.agent_build.build_tools", return_value=self.tool_registry))

    def graph_patch(self, node):
        builder = StateGraph(MessagesState, context_schema=AgentRunContext)
        builder.add_node("work", node)
        builder.add_edge(START, "work")
        builder.add_edge("work", END)
        graph = builder.compile(checkpointer=self.saver)
        def build_loop(*, context, **kwargs):
            loop = object.__new__(AgentLoop)
            loop.context = context
            loop.graph = graph
            return loop
        return patch("app.llm.agent_runner.AgentLoop", side_effect=build_loop)

    async def create_run(self):
        async with self.factory() as db:
            return await agent_run_service.create_run(
                db, user_id=1, thread_id=self.thread.thread_id, metadata={"model_id": self.model_id},
            )

    async def get_run(self, run_id):
        async with self.factory() as db:
            return await agent_run_service.get_run(db, user_id=1, thread_id=self.thread.thread_id, run_id=run_id)

    async def latest_run(self):
        async with self.factory() as db:
            return await db.scalar(select(AgentRun).where(AgentRun.user_id == 1).order_by(AgentRun.id.desc()).limit(1))

    async def post(self, **input):
        return await self.client.post("/api/thread/runs/stream", json={
            "user_id": 1, "thread_id": self.thread.thread_id, "metadata": {"model_id": self.model_id}, **(input or {"input": {"message": "hi"}}),
        })

    async def test_empty_thread_id_creates_thread_and_run(self):
        thread_ids = set()
        for values in ({}, {"thread_id": None}, {"thread_id": ""}):
            with self.subTest(values=values):
                response = await self.client.post("/api/thread/runs/stream", json={
                    "user_id": 1, "metadata": {"model_id": self.model_id}, "input": {"message": "hi"}, **values,
                })
                self.assertEqual(response.status_code, 200)
                output = events(response)
                latest = await self.latest_run()
                thread_id, run_id = latest.thread_id, latest.run_id
                self.assertTrue(thread_id)
                self.assertNotEqual(thread_id, self.thread.thread_id)
                thread_ids.add(thread_id)
                self.assertEqual(output[-1]["status"], "completed")
                async with self.factory() as db:
                    thread = await agent_thread.get_by_thread_id(db, user_id=1, thread_id=thread_id)
                    self.assertIsNotNone(thread)
                    self.assertIsNone(thread.topic)
                    stored = await agent_run.get_by_run_id(db, user_id=1, thread_id=thread_id, run_id=run_id)
                    self.assertEqual(stored.run_metadata, {"model_id": self.model_id})
                self.assertIsNotNone(await self.saver.aget_tuple({"configurable": {"thread_id": thread_id}}))
        self.assertEqual(len(thread_ids), 3)
        async with self.factory() as db:
            self.assertEqual(await db.scalar(select(func.count()).select_from(AgentThread)), 4)
            self.assertEqual(await db.scalar(select(func.count()).select_from(AgentRun)), 3)

    async def test_invalid_new_thread_requests_do_not_create_records(self):
        invalid = [{**values, "resume": None} for values in ({}, {"thread_id": None}, {"thread_id": ""})]
        invalid += [{"input": {"message": "hi"}, "metadata": {}}, {"input": {"message": "hi"}, "user_id": 0}]
        for values in invalid:
            with self.subTest(values=values):
                response = await self.client.post("/api/thread/runs/stream", json={
                    "user_id": 1, "metadata": {"model_id": self.model_id}, **values,
                })
                self.assertEqual(response.status_code, 400)
                self.assertIn("application/json", response.headers["content-type"])
        self.model_constructor.assert_not_called()
        async with self.factory() as db:
            self.assertEqual(await db.scalar(select(func.count()).select_from(AgentThread)), 1)
            self.assertEqual(await db.scalar(select(func.count()).select_from(AgentRun)), 0)

    async def test_new_thread_rolls_back_when_run_insert_fails(self):
        original_create = agent_run.create
        async def invalid_run(db, **fields):
            return await original_create(db, **(fields | {"user_id": None}))
        async with self.factory() as db:
            with patch.object(agent_run, "create", side_effect=invalid_run), self.assertRaises(IntegrityError):
                await agent_run_service.create_run(
                    db, user_id=1, thread_id=None, metadata={"model_id": self.model_id},
                )
            self.assertFalse(db.in_transaction())
            self.assertEqual(await db.scalar(select(func.count()).select_from(AgentThread)), 1)
            self.assertEqual(await db.scalar(select(func.count()).select_from(AgentRun)), 0)

    async def test_failed_execution_keeps_new_thread_and_run(self):
        def fail(state):
            raise RuntimeError("node failed")
        with self.graph_patch(fail), self.assertLogs("app.services.agent_run_service", level="ERROR"):
            response = await self.post(thread_id=None, input={"message": "hi"})
        terminal = events(response)[-1]
        self.assertEqual(terminal["status"], "failed")
        latest = await self.latest_run()
        async with self.factory() as db:
            thread = await agent_thread.get_by_thread_id(db, user_id=1, thread_id=latest.thread_id)
            self.assertIsNotNone(thread)
            stored = await agent_run.get_by_run_id(db, user_id=1, thread_id=thread.thread_id, run_id=latest.run_id)
            self.assertEqual(stored.status, "failed")
            self.assertEqual(stored.error_message, "node failed")

    async def test_defaults_transitions_and_thread_time_commit_together(self):
        run = await self.create_run()
        self.assertEqual((run.status, run.creator_id, run.modifier_id), ("pending", 0, 0))
        self.assertEqual(run.run_metadata, {"model_id": self.model_id})
        self.assertIsNone(run.started_at)
        self.assertIsNone(run.finished_at)
        for status in (RunStatus.RUNNING, RunStatus.COMPLETED):
            async with self.factory() as db:
                changed = await agent_run_service.update_status(
                    db, user_id=1, thread_id=run.thread_id, run_id=run.run_id, status=status,
                )
                self.assertFalse(db.in_transaction())
            async with self.factory() as db:
                thread = await agent_thread.get_by_thread_id(db, user_id=1, thread_id=run.thread_id)
                self.assertEqual(thread.modification_time, changed.modification_time)
                self.assertGreater(thread.modification_time, self.thread.update_time)
        self.assertLessEqual(changed.creation_time, changed.started_at)
        self.assertLessEqual(changed.started_at, changed.finished_at)
        async with self.factory() as db:
            self.assertIsNone(await agent_run_service.update_status(
                db, user_id=1, thread_id=run.thread_id, run_id=run.run_id, status=RunStatus.RUNNING,
            ))
        self.assertEqual((await self.get_run(run.run_id)).status, "completed")

    async def test_status_and_thread_update_roll_back_together(self):
        run = await self.create_run()
        async with self.factory() as db:
            with patch.object(agent_thread, "touch", side_effect=RuntimeError("touch failed")):
                with self.assertRaisesRegex(RuntimeError, "touch failed"):
                    await agent_run_service.update_status(
                        db, user_id=1, thread_id=run.thread_id, run_id=run.run_id, status=RunStatus.RUNNING,
                    )
        stored = await self.get_run(run.run_id)
        self.assertEqual(stored.status, "pending")
        self.assertIsNone(stored.started_at)
        async with self.factory() as db:
            thread = await agent_thread.get_by_thread_id(db, user_id=1, thread_id=run.thread_id)
            self.assertEqual(thread.modification_time, run.creation_time)

    async def test_owner_filtering_and_http_errors_happen_before_sse(self):
        run = await self.create_run()
        response = await self.post()
        self.assertEqual(response.status_code, 409)
        self.assertIn("application/json", response.headers["content-type"])
        for path in (
            f"/api/agent/threads/{run.thread_id}/runs",
            f"/api/agent/threads/{run.thread_id}/runs/{run.run_id}",
        ):
            response = await self.client.get(path, params={"user_id": 2})
            self.assertEqual(response.status_code, 404)
        response = await self.client.post("/api/thread/runs/stream", json={
            "user_id": 2, "metadata": {"model_id": self.model_id}, "thread_id": run.thread_id, "input": {"message": "hi"},
        })
        self.assertEqual(response.status_code, 404)
        response = await self.post(thread_id="nonexistent", input={"message": "hi"})
        self.assertEqual(response.status_code, 404)
        async with self.factory() as db:
            self.assertEqual(await db.scalar(select(func.count()).select_from(AgentThread)), 1)
            self.assertIsNone(await agent_run.get_by_run_id(db, user_id=2, thread_id=run.thread_id, run_id=run.run_id))
        async with self.factory() as db:
            self.assertIsNone(await agent_run_service.update_status(
                db, user_id=2, thread_id=run.thread_id, run_id=run.run_id, status=RunStatus.CANCELLED,
            ))

    async def test_http_input_validation_and_thread_routes(self):
        for input in (
            {}, {"input": {"message": "hi"}, "resume": True},
            {"input": {"message": "hi"}, "metadata": {"model_id": 0}},
            {"input": {"message": "hi"}, "metadata": {"model_id": self.model_id, "think_effort": "invalid"}},
        ):
            response = await self.client.post("/api/thread/runs/stream", json={
                "user_id": 1, "metadata": {"model_id": self.model_id}, "thread_id": self.thread.thread_id, **input,
            })
            self.assertEqual(response.status_code, 400)
        created = await self.client.post("/api/thread", json={"user_id": 1, "topic": "new thread"})
        self.assertEqual(created.status_code, 200)
        new_id = created.json()["data"]["thread_id"]
        listed = await self.client.get("/api/thread/list", params={"user_id": 1, "limit": 1})
        self.assertEqual([row["thread_id"] for row in listed.json()["data"]], [new_id])
        async with self.factory() as db:
            self.assertEqual(await db.scalar(select(func.count()).select_from(AgentRun)), 0)

    async def test_concurrent_requests_and_database_constraints(self):
        async def create():
            try:
                return await self.create_run()
            except ConflictException as exc:
                return exc
        results = await asyncio.gather(create(), create())
        self.assertEqual(sum(isinstance(result, AgentRun) for result in results), 1)
        self.assertEqual(sum(isinstance(result, ConflictException) for result in results), 1)
        async with self.factory() as db:
            with self.assertRaises(IntegrityError):
                async with db.begin():
                    await agent_run.create(db, user_id=1, thread_id=self.thread.thread_id, run_id="duplicate", metadata={})
        async with self.factory() as db:
            with self.assertRaises(IntegrityError):
                async with db.begin():
                    db.add(AgentRun(user_id=1, thread_id=self.thread.thread_id, run_id="bad-status", run_metadata={}, status="invalid"))
                    await db.flush()

    async def test_completed_run_and_paginated_queries(self):
        def work(state):
            return {"messages": [AIMessage(content="done")]}
        with self.graph_patch(work):
            first = await self.post()
            first_id = (await self.latest_run()).run_id
            second = await self.post(input={"message": "again"})
            second_id = (await self.latest_run()).run_id
        first_events, second_events = events(first), events(second)
        self.assertEqual(first_events[-1]["status"], "completed")
        self.assertNotIn("interrupt", [event["type"] for event in first_events + second_events])
        self.assertNotEqual(first_id, second_id)
        response = await self.client.get(f"/api/agent/threads/{self.thread.thread_id}/runs", params={"user_id": 1})
        self.assertEqual([row["run_id"] for row in response.json()["data"]], [second_id, first_id])
        response = await self.client.get(f"/api/agent/threads/{self.thread.thread_id}/runs", params={"user_id": 1, "limit": 1, "offset": 1})
        self.assertEqual([row["run_id"] for row in response.json()["data"]], [first_id])
        response = await self.client.get(f"/api/agent/threads/{self.thread.thread_id}/runs/{first_id}", params={"user_id": 1})
        self.assertEqual(response.json()["data"]["status"], "completed")
        async with self.factory() as db:
            self.assertEqual(await db.scalar(select(func.count()).select_from(AgentThread)), 1)

    async def test_graph_exception_is_failed_and_does_not_emit_done(self):
        def fail(state):
            raise RuntimeError("node failed")
        with self.graph_patch(fail), self.assertLogs("app.services.agent_run_service", level="ERROR"):
            response = await self.post()
        output = events(response)
        self.assertEqual(output[-1]["status"], "failed")
        self.assertNotIn("done", [event["type"] for event in output])
        self.assertNotIn("interrupt", [event["type"] for event in output])
        stored = await self.latest_run()
        self.assertEqual(stored.error_message, "node failed")
        self.assertIsNotNone(stored.finished_at)

    async def test_main_agent_loop_streams_tokens_and_completes(self):
        def create_runner(**kwargs):
            self.model_constructor.assert_called_once_with(
                model="test-model", base_url="https://example.test/v1", api_key="test-key", streaming=True,
                reasoning_effort=None,
            )
            self.tools_builder.assert_called_once()
            self.assertEqual(self.tools_builder.call_args.kwargs, {"user_id": 1, "model": self.model_constructor.return_value})
            self.assertIs(kwargs["model"], self.model_constructor.return_value)
            self.assertIs(kwargs["tool_registry"], self.tool_registry)
            return AgentRunner(**kwargs)
        with patch("app.services.agent_run_service.AgentRunner", side_effect=create_runner):
            response = await self.post()
        output = events(response)
        self.assertEqual("".join(event["content"] for event in output if event["type"] == "token"), "hello")
        self.assertEqual(output[-1]["status"], "completed")
        self.assertEqual((await self.latest_run()).run_metadata, {"model_id": self.model_id})

    async def test_metadata_is_saved_and_effort_reaches_model_constructor(self):
        for effort in ("high", None):
            with self.subTest(effort=effort):
                self.model_constructor.reset_mock()
                metadata = {"model_id": self.model_id, "think_effort": effort}
                response = await self.post(input={"message": "hi"}, metadata=metadata)
                output = events(response)
                self.assertEqual(output[-1]["status"], "completed")
                self.model_constructor.assert_called_once_with(
                    model="test-model", base_url="https://example.test/v1", api_key="test-key",
                    streaming=True, reasoning_effort=effort,
                )
                stored = await self.latest_run()
                self.assertEqual(stored.run_metadata, metadata)
                detail = await self.client.get(
                    f"/api/agent/threads/{stored.thread_id}/runs/{stored.run_id}", params={"user_id": 1},
                )
                self.assertEqual(detail.json()["data"]["metadata"], metadata)
                self.assertNotIn("input", detail.json()["data"])

    async def test_existing_input_column_migration_keeps_only_model_metadata(self):
        run = await self.create_run()
        metadata = {"model_id": self.model_id, "think_effort": "medium"}
        async with self.factory.begin() as db:
            await db.execute(text("ALTER TABLE agent_run RENAME COLUMN metadata TO input"))
            await db.execute(text("UPDATE agent_run SET input = CAST(:payload AS JSON) WHERE run_id = :run_id"), {
                "payload": json.dumps({"metadata": metadata, "message": "old message"}), "run_id": run.run_id,
            })
        migration = Path(__file__).resolve().parents[1] / "migrations/20261020_rename_agent_run_input_to_metadata.sql"
        async with await AsyncConnection.connect(
            make_conninfo(TEST_DATABASE_URL, options=f"-csearch_path={self.schema}"), autocommit=True,
        ) as connection:
            await connection.execute(migration.read_text(encoding="utf-8"))
        stored = await self.get_run(run.run_id)
        self.assertEqual(stored.run_metadata, metadata)
        self.assertEqual((stored.id, stored.run_id, stored.creation_time), (run.id, run.run_id, run.creation_time))
        async with self.factory() as db:
            columns = set(await db.scalars(text(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = :schema AND table_name = 'agent_run'"
            ), {"schema": self.schema}))
        self.assertIn("metadata", columns)
        self.assertNotIn("input", columns)

    async def test_model_lookup_and_build_failures_fail_run_before_runner_creation(self):
        async with self.factory() as db:
            foreign = await llm_setting_service.create_setting(
                db, user_id=2, provider="unused-provider", base_url="https://example.test/v1",
                model="foreign", api_key="foreign-key",
            )
        cases = (
            (self.model_id + 1000, None, None, "LLM 配置不存在"),
            (foreign.id, None, None, "LLM 配置不存在"),
            (self.model_id, RuntimeError("model construction failed"), None, "model construction failed"),
            (self.model_id, None, RuntimeError("tool construction failed"), "tool construction failed"),
        )
        for model_id, model_error, tool_error, expected_error in cases:
            with self.subTest(error=expected_error, model_id=model_id):
                self.model_constructor.side_effect = model_error
                self.tools_builder.side_effect = tool_error
                with patch("app.services.agent_run_service.AgentRunner") as runner, self.assertLogs(
                    "app.services.agent_run_service", level="ERROR",
                ):
                    response = await self.post(input={"message": "hi"}, metadata={"model_id": model_id})
                runner.assert_not_called()
                output = events(response)
                self.assertEqual(output[-1]["status"], "failed")
                self.assertNotIn("done", [event["type"] for event in output])
                stored = await self.latest_run()
                self.assertEqual(stored.status, "failed")
                self.assertEqual(stored.error_message, expected_error)

    async def test_history_uses_checkpoint_and_checks_thread_owner(self):
        params = {"user_id": 1, "thread_id": self.thread.thread_id}
        response = await self.client.get("/api/agent/history", params=params)
        self.assertEqual(response.json()["data"], [])
        with self.graph_patch(lambda state: {"messages": [AIMessage(content="checkpoint reply")]}):
            await self.post()
        response = await self.client.get("/api/agent/history", params=params)
        self.assertEqual(response.status_code, 200)
        self.assertEqual([message["content"] for message in response.json()["data"]["messages"]], ["hi", "checkpoint reply"])
        response = await self.client.get("/api/agent/history", params=params | {"user_id": 2})
        self.assertEqual(response.status_code, 404)

    async def test_real_interrupt_and_resume_create_separate_runs(self):
        def approval(state):
            answer = interrupt({"question": "approve?"})
            return {"messages": [AIMessage(content=str(answer))]}
        with self.graph_patch(approval):
            paused_events = events(await self.post())
            paused_run = await self.latest_run()
            self.assertEqual(
                [event["type"] for event in paused_events if event["type"] in {"interrupt", "done"}],
                ["interrupt", "done"],
            )
            paused = paused_events[-1]
            self.assertEqual(paused_events[-2], {**paused, "type": "interrupt"})
            self.assertEqual(paused["status"], "interrupted")
            self.assertTrue(paused["interrupts"][0]["id"])
            self.assertEqual(paused["interrupts"][0]["value"], {"question": "approve?"})
            resumed_events = events(await self.post(resume=True, metadata={"model_id": self.model_id, "think_effort": "low"}))
            resumed_run = await self.latest_run()
            self.assertNotIn("interrupt", [event["type"] for event in resumed_events])
            resumed = resumed_events[-1]
            self.assertEqual(self.model_constructor.call_args.kwargs["reasoning_effort"], "low")
        self.assertEqual(resumed["status"], "completed")
        self.assertNotEqual(paused_run.run_id, resumed_run.run_id)
        self.assertEqual(paused_run.status, "interrupted")
        self.assertEqual(resumed_run.run_metadata, {
            "model_id": self.model_id, "think_effort": "low",
        })

    async def test_disconnect_after_interrupt_preserves_committed_status(self):
        def approval(state):
            interrupt({"question": "approve?"})
            return {}
        run = await self.create_run()
        response = RunStreamingResponse(run, self.factory, self.saver, None, input={"message": "hi"})
        sent_types = []
        async def receive():
            await asyncio.Event().wait()
        async def send(message):
            body = message.get("body", b"")
            if not body:
                return
            event = json.loads(body[6:])
            sent_types.append(event["type"])
            if event["type"] == "interrupt":
                stored = await self.get_run(run.run_id)
                self.assertEqual(stored.status, "interrupted")
                self.assertIsNotNone(stored.finished_at)
            if event["type"] == "done":
                raise OSError("client disconnected after interrupt")
        with self.graph_patch(approval):
            with self.assertRaises(ClientDisconnect):
                await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        self.assertEqual(sent_types[-2:], ["interrupt", "done"])
        self.assertEqual((await self.get_run(run.run_id)).status, "interrupted")

    async def test_disconnect_during_graph_execution_closes_graph_and_cancels_run(self):
        for spec in ("2.3", "2.4"):
            with self.subTest(asgi=spec):
                entered, closed = asyncio.Event(), asyncio.Event()
                async def blocking(state):
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        closed.set()
                run = await self.create_run()
                response = RunStreamingResponse(run, self.factory, self.saver, None, input={"message": "hi"})
                async def receive():
                    await entered.wait()
                    return {"type": "http.disconnect"}
                async def send(message):
                    pass
                with self.graph_patch(blocking):
                    task = asyncio.create_task(response({"type": "http", "asgi": {"spec_version": spec}}, receive, send))
                    await asyncio.wait_for(entered.wait(), 5)
                    if spec == "2.4":
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await asyncio.wait_for(task, 5)
                    else:
                        await asyncio.wait_for(task, 5)
                self.assertTrue(closed.is_set())
                stored = await self.get_run(run.run_id)
                self.assertEqual(stored.status, "cancelled")
                self.assertIsNotNone(stored.finished_at)

    async def test_send_failure_cancels_pending_and_running_but_preserves_completed(self):
        for fail_on in ("headers", "running", "done"):
            with self.subTest(fail_on=fail_on):
                run = await self.create_run()
                response = RunStreamingResponse(run, self.factory, self.saver, None, input={"message": "hi"})
                async def receive():
                    await asyncio.Event().wait()
                async def send(message):
                    body = message.get("body", b"")
                    if (fail_on == "headers" and message["type"] == "http.response.start") or (
                        fail_on == "running" and b'"status": "running"' in body
                    ) or (fail_on == "done" and b'"type": "done"' in body):
                        raise OSError("client disconnected")
                with self.graph_patch(lambda state: {"messages": [AIMessage(content="ok")]}):
                    with self.assertRaises(ClientDisconnect):
                        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
                stored = await self.get_run(run.run_id)
                self.assertEqual(stored.status, "completed" if fail_on == "done" else "cancelled")
                if fail_on == "headers":
                    self.assertIsNone(stored.started_at)

    async def test_thread_deletion_rejects_active_run_then_removes_runs_and_checkpoints(self):
        run = await self.create_run()
        response = await self.client.delete(f"/api/thread/{run.thread_id}", params={"user_id": 1})
        self.assertEqual(response.status_code, 409)
        async with self.factory() as db:
            await agent_run_service.update_status(db, user_id=1, thread_id=run.thread_id, run_id=run.run_id, status=RunStatus.CANCELLED)
        with self.graph_patch(lambda state: {"messages": [AIMessage(content="ok")]}):
            await self.post()
        config = {"configurable": {"thread_id": run.thread_id}}
        self.assertIsNotNone(await self.saver.aget_tuple(config))
        response = await self.client.delete(f"/api/thread/{run.thread_id}", params={"user_id": 1})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(await self.saver.aget_tuple(config))
        async with self.factory() as db:
            self.assertEqual(await db.scalar(select(func.count()).select_from(AgentRun)), 0)
            self.assertEqual(await db.scalar(select(func.count()).select_from(AgentThread)), 0)


if __name__ == "__main__":
    unittest.main()
