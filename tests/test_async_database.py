from __future__ import annotations

import json
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import grpc
import httpx
from fastapi import FastAPI
from sqlalchemy import inspect, select

from app.config import settings
from app.db.base import Base
from app.db.session import create_database, get_db
from app.grpc.services import add_services
from app.db.models import MemoryConsolidationCursor, MemoryEpisode, MemoryFact
from app.db.models import SystemSetting
from app.db.repositories import memory_consolidation_cursor, memory_fact
from app.memory.episodic.service import EpisodeService
from app.memory.errors import MemoryNotFoundError
from app.memory.schemas import EpisodeCandidate, FactCandidate
from app.memory.semantic.service import FactService
from app.protos.agent.v1 import agent_pb2, agent_pb2_grpc


class AsyncDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        with patch.object(
            settings,
            "DATABASE_URL",
            "sqlite+aiosqlite:///:memory:",
        ), patch.object(
            settings,
            "SQL_ECHO",
            False,
        ):
            self.engine, self.factory = create_database()
        async with self.engine.begin() as connection:
            for table in (
                SystemSetting.__table__,
                MemoryFact.__table__,
                MemoryEpisode.__table__,
                MemoryConsolidationCursor.__table__,
            ):
                await connection.run_sync(table.create)

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def test_only_retained_tables_are_created(self):
        expected = {
            "config_setting", "memory_facts",
            "memory_episodes", "memory_consolidation_cursors",
        }
        self.assertEqual(set(Base.metadata.tables), expected | {
            "agent_session", "agent_message", "agent_summary",
        })
        async with self.engine.connect() as connection:
            actual = await connection.run_sync(
                lambda sync_connection: inspect(sync_connection).get_table_names(),
            )
        self.assertEqual(set(actual), expected)

    async def test_commit_keeps_loaded_attributes_and_sessions_are_independent(self):
        async with self.factory() as first, self.factory() as second:
            self.assertIsNot(first, second)
            fact = MemoryFact(
                user_id="1", subject="topic", content="saved",
                dedupe_key="saved", source="explicit",
            )
            first.add(fact)
            await first.commit()
            self.assertEqual(fact.content, "saved")
            stored = await second.scalar(select(MemoryFact))
            self.assertEqual(stored.content, "saved")

    async def test_http_dependency_rolls_back_uncommitted_changes(self):
        request = SimpleNamespace(app=SimpleNamespace(
            state=SimpleNamespace(session_factory=self.factory),
        ))
        dependency = get_db(request)
        session = await anext(dependency)
        session.add(MemoryFact(
            user_id="1", subject="topic", content="uncommitted",
            dedupe_key="uncommitted", source="explicit",
        ))
        await session.flush()
        await dependency.aclose()
        async with self.factory() as check:
            self.assertIsNone(await check.scalar(select(MemoryFact)))

    async def test_memory_crud_preserves_deduplication_and_user_scope(self):
        async with self.factory() as session:
            owner = FactService(session, user_id="owner")
            other = FactService(session, user_id="other")
            fact, created = await owner.remember_explicit(
                subject="Topic", content="original", source_conversation_id="chat",
            )
            fact_id = fact.id
            self.assertTrue(created)
            duplicate, created = await owner.remember_explicit(
                subject="topic", content="ORIGINAL", source_conversation_id="chat",
            )
            self.assertEqual(duplicate.id, fact_id)
            self.assertFalse(created)
            self.assertEqual(await other.search_active(query="original"), [])
            with self.assertRaises(MemoryNotFoundError):
                await other.correct(fact_id=fact_id, subject="topic", content="foreign")
            changed = await owner.correct(
                fact_id=fact_id, subject="topic", content="updated",
            )
            self.assertIsNotNone(changed.updated_at)
            self.assertEqual((await owner.search_active(query="updated"))[0].id, fact_id)
            await owner.forget(fact_id=fact_id)
            self.assertEqual(await owner.get_active_by_ids([fact_id]), [])
            restored, created = await owner.remember_explicit(
                subject="topic", content="updated", source_conversation_id="chat",
            )
            self.assertTrue(created)
            self.assertEqual(restored.id, fact_id)

    async def test_service_rolls_back_failed_write_and_session_remains_usable(self):
        original_update = memory_fact.update

        async def fail_after_update(
            db,
            fact,
            **fields,
        ):
            await original_update(
                db,
                fact,
                **fields,
            )
            raise RuntimeError("write failed")

        async with self.factory() as session:
            service = FactService(session, user_id="owner")
            fact, _ = await service.remember_explicit(
                subject="topic",
                content="original",
                source_conversation_id=None,
            )
            fact_id = fact.id
            with patch.object(
                memory_fact,
                "update",
                side_effect=fail_after_update,
            ):
                with self.assertRaisesRegex(RuntimeError, "write failed"):
                    await service.correct(
                        fact_id=fact_id,
                        subject="topic",
                        content="changed",
                    )
            self.assertFalse(session.in_transaction())
            self.assertEqual(
                (await service.get_active_by_ids([fact_id]))[0].content, "original"
            )

    async def test_staged_memories_and_cursor_share_one_transaction(self):
        async with self.factory() as session:
            facts = FactService(session, user_id="owner")
            episodes = EpisodeService(session, user_id="owner")
            with self.assertRaisesRegex(RuntimeError, "abort"):
                async with session.begin():
                    await facts.stage_consolidated_candidate(
                        candidate=FactCandidate(subject="topic", content="pending"),
                        source_conversation_id="chat",
                        source_message_ids=[1],
                    )
                    await memory_consolidation_cursor.advance(
                        session,
                        user_id="owner",
                        conversation_id="chat",
                        last_assistant_message_id=1,
                    )
                    raise RuntimeError("abort")
            async with session.begin():
                self.assertIsNone(await session.scalar(select(MemoryFact)))
                self.assertIsNone(
                    await memory_consolidation_cursor.get(
                        session,
                        user_id="owner",
                        conversation_id="chat",
                    )
                )
                episode = await episodes.stage_consolidated_candidate(
                    candidate=EpisodeCandidate(
                        happened_at=date(
                            2026,
                            10,
                            15,
                        ),
                        summary="summary",
                        source_assistant_message_id=2,
                    ),
                    source_conversation_id="chat",
                )
                episode_id = episode.id
                await memory_consolidation_cursor.advance(
                    session,
                    user_id="owner",
                    conversation_id="chat",
                    last_assistant_message_id=2,
                )
            self.assertEqual(
                await episodes.get_active_by_ids_for_conversation(
                    episode_ids=[episode_id],
                    conversation_id="another-chat",
                ),
                [],
            )
            self.assertEqual(
                (await episodes.list_active_for_conversation(conversation_id="chat"))[0].id,
                episode_id,
            )
            async with session.begin():
                cursor = await memory_consolidation_cursor.get(
                    session,
                    user_id="owner",
                    conversation_id="chat",
                )
                self.assertEqual(cursor.last_assistant_message_id, 2)

    async def test_mysql_sync_url_selects_an_async_driver_without_connecting(self):
        with patch.object(settings, "DATABASE_URL", "mysql+pymysql://user:password@localhost/example"):
            engine, factory = create_database()
        try:
            self.assertEqual(engine.url.drivername, "mysql+aiomysql")
            self.assertFalse(factory.kw["expire_on_commit"])
        finally:
            await engine.dispose()

    async def test_settings_grpc_persists_limits_and_retires_old_rpcs(self):
        async with self.factory() as session:
            session.add(SystemSetting(
                id=1, key="source_limits",
                value=json.dumps({"arxiv": 1, "dblp": 1, "google_scholar": 1}),
            ))
            await session.commit()
        server = grpc.aio.server()
        add_services(server, self.factory)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                stub = agent_pb2_grpc.SettingsServiceStub(channel)
                updated = await stub.UpdateSourceLimits(
                    agent_pb2.UpdateSourceLimitsRequest(
                        arxiv=2, dblp=2, google_scholar=2,
                    ), timeout=5,
                )
                self.assertEqual(updated.google_scholar, 2)
                stored = await stub.GetSourceLimits(
                    agent_pb2.GetSourceLimitsRequest(), timeout=5,
                )
                self.assertEqual(stored.arxiv, 2)
                with self.assertRaises(grpc.aio.AioRpcError) as invalid:
                    await stub.UpdateSourceLimits(
                        agent_pb2.UpdateSourceLimitsRequest(arxiv=0, dblp=1, google_scholar=1),
                        timeout=5,
                    )
                self.assertEqual(invalid.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
                with self.assertRaises(grpc.aio.AioRpcError) as profile:
                    await stub.ListLlmProfiles(agent_pb2.ListLlmProfilesRequest(), timeout=5)
                self.assertEqual(profile.exception.code(), grpc.StatusCode.UNIMPLEMENTED)
                agent = agent_pb2_grpc.AgentServiceStub(channel)
                with self.assertRaises(grpc.aio.AioRpcError) as retired:
                    await agent.StartRun(agent_pb2.StartRunRequest(), timeout=5)
                self.assertEqual(retired.exception.code(), grpc.StatusCode.UNIMPLEMENTED)
        finally:
            await server.stop(0)

    async def test_http_settings_use_async_sessions(self):
        from app.api.settings import router

        app = FastAPI()
        app.state.session_factory = self.factory
        app.include_router(router)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.get("/api/settings/source-limits")
            self.assertEqual(response.status_code, 200)
            self.assertIn("limits", response.json()["data"])
            removed = await client.get("/api/settings/llm-profiles")
            self.assertEqual(removed.status_code, 404)

    async def test_lifespan_releases_database_even_when_startup_fails(self):
        from app import main

        engine_spy = Mock(wraps=self.engine)
        engine_spy.dispose = AsyncMock(wraps=self.engine.dispose)
        index = Mock(ensure_collections=AsyncMock(side_effect=RuntimeError("index unavailable")))
        with patch.object(main, "create_database", return_value=(engine_spy, self.factory)), patch.object(
            main, "get_memory_index_synchronizer", return_value=index,
        ), patch.object(main, "close_index_resources", new_callable=AsyncMock) as close_index:
            with self.assertRaisesRegex(RuntimeError, "index unavailable"):
                async with main.lifespan(FastAPI()):
                    self.fail("startup should fail")
        engine_spy.dispose.assert_awaited_once()
        close_index.assert_awaited_once()

    async def test_lifespan_stops_services_and_disposes_database_on_shutdown(self):
        from app import main

        engine_spy = Mock(wraps=self.engine)
        engine_spy.dispose = AsyncMock(wraps=self.engine.dispose)
        server = Mock(start=AsyncMock(), stop=AsyncMock())
        nacos = Mock(start=AsyncMock(), stop=AsyncMock())
        index = Mock(ensure_collections=AsyncMock())
        app = FastAPI()
        with patch.object(main, "create_database", return_value=(engine_spy, self.factory)), patch.object(
            main, "get_memory_index_synchronizer", return_value=index,
        ), patch.object(main, "InternalGrpcServer", return_value=server), patch.object(
            main, "nacos_registry", nacos,
        ), patch.object(main, "close_index_resources", new_callable=AsyncMock), patch.object(
            main, "close_task_rag_batch_runner", new_callable=AsyncMock,
        ), patch.object(main.paper_service_grpc_channel_pool, "close", new_callable=AsyncMock):
            async with main.lifespan(app):
                self.assertIs(app.state.session_factory, self.factory)
                engine_spy.dispose.assert_not_awaited()
        server.stop.assert_awaited_once()
        nacos.stop.assert_awaited_once()
        engine_spy.dispose.assert_awaited_once()
