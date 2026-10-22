from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import grpc
import httpx
from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, inspect, select

from app.common.exceptions import NotFoundException
from app.database import Base, get_db
from app.grpc.services import add_services
from app.db.models import LlmSetting, SystemSetting
from app.db.repositories import llm_setting, system_setting
from app.protos.agent.v1 import agent_pb2, agent_pb2_grpc
from app.services import llm_setting_service
from app.services.llm_setting_service import LlmConfig
from tests.test_postgres_models import PostgresDatabaseTestCase


class AsyncDatabaseTests(PostgresDatabaseTestCase):
    async def test_all_business_tables_are_created(self):
        expected = {
            "config_setting", "agent_thread", "llm_setting", "agent_run",
        }
        self.assertEqual(set(Base.metadata.tables), expected)
        async with self.engine.connect() as connection:
            actual = await connection.run_sync(
                lambda sync_connection: inspect(sync_connection).get_table_names(
                    schema=self.schema
                ),
            )
        self.assertEqual(set(actual), expected)

    async def test_llm_service_commits_crud_and_filters_settings(self):
        fields = dict(provider="openai", base_url="https://example.test/v1", model="large", api_key="test-key")
        async with self.factory() as db:
            large = await llm_setting_service.create_setting(db, user_id=1, **fields, operator_id=7)
            small = await llm_setting_service.create_setting(db, user_id=1, **fields, operator_id=8, is_small_model=True)
            large_id, small_id = large.id, small.id
            self.assertFalse(db.in_transaction())
            self.assertFalse(large.is_small_model)
            self.assertEqual(large.creator_id, 0)
            self.assertEqual([r.id for r in await llm_setting_service.list_settings(db, user_id=1)], [small_id, large_id])
            self.assertEqual([r.id for r in await llm_setting_service.list_settings(db, user_id=1, is_small_model=False)], [large_id])
            self.assertEqual([r.id for r in await llm_setting_service.list_settings(db, user_id=1, is_small_model=True)], [small_id])
            self.assertEqual([r.id for r in await llm_setting_service.list_settings(db, user_id=1, limit=1, offset=1)], [large_id])
            changed = await llm_setting_service.update_setting(
                db, large_id, user_id=1, provider="deepseek", base_url="https://other.test/v1",
                model="small", api_key="new-key", is_small_model=True, operator_id=9,
            )
            self.assertEqual(changed.creator_id, 0)
            self.assertEqual(changed.modifier_id, 9)
        # A new session must see committed writes, not just the original identity map.
        async with self.factory() as db:
            stored = await llm_setting_service.get_setting(db, large_id, user_id=1)
            self.assertEqual((stored.provider, stored.base_url, stored.model, stored.api_key, stored.is_small_model),
                             ("deepseek", "https://other.test/v1", "small", "new-key", True))
            self.assertTrue(await llm_setting_service.delete_setting(db, small_id, user_id=1))
            self.assertFalse(await llm_setting_service.delete_setting(db, small_id, user_id=1))
            self.assertIsNone(await llm_setting_service.get_setting(db, small_id, user_id=1))
            with self.assertRaisesRegex(ValueError, "not found"):
                await llm_setting_service.update_setting(db, small_id, user_id=1, **fields, is_small_model=False, operator_id=9)
            self.assertFalse(db.in_transaction())
        async with self.factory() as db:
            self.assertEqual([r.id for r in await llm_setting_service.list_settings(db, user_id=1)], [large_id])

    async def test_llm_service_returns_config_and_reports_missing_setting(self):
        async with self.factory() as db:
            setting = await llm_setting_service.create_setting(
                db, user_id=1, provider="deepseek", base_url="https://example.test/v1",
                model="test-model", api_key="test-key", operator_id=7,
            )
        async with self.factory() as db:
            config = await llm_setting_service.get_llm_config(db, model_id=setting.id, user_id=1)
            self.assertEqual(config, LlmConfig(
                provider="deepseek", base_url="https://example.test/v1",
                model="test-model", api_key="test-key",
            ))
            self.assertFalse(db.in_transaction())
            await llm_setting_service.delete_setting(db, setting.id, user_id=1)
            with self.assertRaises(NotFoundException):
                await llm_setting_service.get_llm_config(db, model_id=setting.id, user_id=1)
            self.assertFalse(db.in_transaction())

    async def test_llm_settings_are_scoped_to_owner_not_operator(self):
        fields = dict(provider="openai", base_url="https://example.test/v1", model="original", api_key="owner-key")
        async with self.factory() as db:
            owned = await llm_setting_service.create_setting(db, user_id=11, **fields, operator_id=22)
            small = await llm_setting_service.create_setting(db, user_id=11, **fields, operator_id=22, is_small_model=True)
            foreign = await llm_setting_service.create_setting(db, user_id=22, **fields, operator_id=11, is_small_model=True)
            owned_id, small_id, foreign_id = owned.id, small.id, foreign.id

            # Even a row already in this session's identity map must be filtered by SQL.
            self.assertIsNone(await llm_setting_service.get_setting(db, owned_id, user_id=22))
            self.assertIsNone(await llm_setting_service.get_setting(db, owned_id, user_id=0))
            self.assertEqual((owned.creator_id, small.creator_id, foreign.creator_id), (0, 0, 0))
            self.assertEqual([r.id for r in await llm_setting_service.list_settings(db, user_id=11)], [small_id, owned_id])
            self.assertEqual([r.id for r in await llm_setting_service.list_settings(db, user_id=22)], [foreign_id])
            self.assertEqual([r.id for r in await llm_setting_service.list_settings(db, user_id=11, is_small_model=True)], [small_id])
            self.assertEqual([r.id for r in await llm_setting_service.list_settings(db, user_id=11, is_small_model=False)], [owned_id])
            self.assertEqual([r.id for r in await llm_setting_service.list_settings(db, user_id=11, limit=1, offset=1)], [owned_id])
            self.assertEqual(await llm_setting_service.list_settings(db, user_id=33), [])
            with self.assertRaises(NotFoundException):
                await llm_setting_service.get_llm_config(db, owned_id, user_id=22)
            with self.assertRaises(NotFoundException):
                await llm_setting_service.get_llm_config(db, foreign_id, user_id=11)
            with self.assertRaisesRegex(ValueError, "not found"):
                await llm_setting_service.update_setting(
                    db, owned_id, user_id=22, **{**fields, "api_key": "foreign-key"},
                    is_small_model=True, operator_id=11,
                )
            self.assertFalse(await llm_setting_service.delete_setting(db, owned_id, user_id=22))

            stored = await llm_setting_service.get_setting(db, owned_id, user_id=11)
            self.assertEqual((stored.user_id, stored.creator_id, stored.modifier_id, stored.api_key),
                             (11, 0, 22, "owner-key"))
            updated = await llm_setting_service.update_setting(
                db, owned_id, user_id=11, **fields, is_small_model=False, operator_id=33,
            )
            self.assertEqual((updated.user_id, updated.creator_id, updated.modifier_id), (11, 0, 33))
            config = await llm_setting_service.get_llm_config(db, owned_id, user_id=11)
            self.assertEqual(config.api_key, "owner-key")
            self.assertTrue(await llm_setting_service.delete_setting(db, owned_id, user_id=11))

    async def test_llm_service_rolls_back_failed_update(self):
        fields = dict(provider="openai", base_url="https://example.test/v1", model="original", api_key="test-key")
        original_update = llm_setting.update

        async def fail_after_flush(db, model_id, **values):
            await original_update(db, model_id, **values)
            raise RuntimeError("write failed")

        async with self.factory() as db:
            setting = await llm_setting_service.create_setting(db, user_id=1, **fields, operator_id=7)
            model_id = setting.id
            with patch.object(llm_setting, "update", side_effect=fail_after_flush):
                with self.assertRaisesRegex(RuntimeError, "write failed"):
                    await llm_setting_service.update_setting(
                        db, model_id, user_id=1, **{**fields, "model": "changed"},
                        is_small_model=True, operator_id=9,
                    )
            self.assertFalse(db.in_transaction())
            stored = await llm_setting_service.get_setting(db, model_id, user_id=1)
            self.assertEqual((stored.model, stored.is_small_model, stored.modifier_id), ("original", False, 7))
        async with self.factory() as db:
            stored = await llm_setting_service.get_setting(db, model_id, user_id=1)
            self.assertEqual(stored.model, "original")

    async def test_llm_crud_filter_pagination_and_rollback(self):
        fields = dict(provider="openai", base_url="https://example.test/v1", model="large", api_key="test-key")
        async with self.factory.begin() as db:
            large = await llm_setting.create(db, user_id=1, **fields, operator_id=7)
            small = await llm_setting.create(db, user_id=1, **fields, operator_id=8, is_small_model=True)
            large_id, small_id = large.id, small.id
            self.assertFalse(large.is_small_model)
            self.assertGreater(small_id, large_id)
            self.assertEqual(large.creator_id, 0)
            self.assertIsNotNone(large.creation_time.tzinfo)
        async with self.factory.begin() as db:
            self.assertEqual([r.id for r in await llm_setting.list_settings(db, user_id=1)], [small_id, large_id])
            self.assertEqual([r.id for r in await llm_setting.list_settings(db, user_id=1, is_small_model=False)], [large_id])
            self.assertEqual([r.id for r in await llm_setting.list_settings(db, user_id=1, is_small_model=True)], [small_id])
            self.assertEqual([r.id for r in await llm_setting.list_settings(db, user_id=1, limit=1, offset=1)], [large_id])
            changed = await llm_setting.update(
                db, large_id, user_id=1, provider="deepseek", base_url="https://other.test/v1",
                model="small", api_key="new-key", is_small_model=True, operator_id=9,
            )
            self.assertEqual(changed.creator_id, 0)
            self.assertEqual(changed.modifier_id, 9)
        async with self.factory.begin() as db:
            changed = await llm_setting.get_by_id(db, large_id, user_id=1)
            self.assertEqual((changed.provider, changed.base_url, changed.model, changed.api_key, changed.is_small_model),
                             ("deepseek", "https://other.test/v1", "small", "new-key", True))
            self.assertTrue(await llm_setting.delete(db, small_id, user_id=1))
            self.assertFalse(await llm_setting.delete(db, small_id, user_id=1))
            self.assertIsNone(await llm_setting.get_by_id(db, small_id, user_id=1))
            with self.assertRaisesRegex(ValueError, "not found"):
                await llm_setting.update(db, small_id, user_id=1, **fields, is_small_model=False, operator_id=9)
        async with self.factory() as db:
            await llm_setting.create(db, user_id=1, **fields, operator_id=7)
            await db.rollback()
        async with self.factory() as db:
            self.assertEqual(await db.scalar(select(func.count()).select_from(LlmSetting)), 1)

    async def test_postgres_defaults_and_automatic_update_time(self):
        old_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
        async with self.factory.begin() as db:
            await system_setting.set_value(db, "key", "first")
            row = await system_setting.get(db, "key")
            row.modification_time = old_time
            # Exercise server defaults as well as the Python Boolean default.
            await db.execute(LlmSetting.__table__.insert().values(
                user_id=1,
                provider="openai", base_url="https://example.test", model="model", api_key="key",
            ))
        async with self.factory.begin() as db:
            await system_setting.set_value(db, "key", "second")
        async with self.factory() as db:
            row = await system_setting.get(db, "key")
            self.assertGreater(row.modification_time, old_time)
            self.assertIsNotNone(row.creation_time.tzinfo)
            llm = await db.scalar(select(LlmSetting))
            self.assertFalse(llm.is_small_model)
            self.assertEqual((llm.creator_id, llm.modifier_id), (0, 0))

    async def test_commit_keeps_loaded_attributes_and_sessions_are_independent(self):
        async with self.factory() as first, self.factory() as second:
            self.assertIsNot(first, second)
            setting = SystemSetting(key="saved", value="saved")
            first.add(setting)
            await first.commit()
            self.assertEqual(setting.value, "saved")
            stored = await second.scalar(select(SystemSetting))
            self.assertEqual(stored.value, "saved")

    async def test_http_dependency_rolls_back_uncommitted_changes(self):
        request = SimpleNamespace(app=SimpleNamespace(
            state=SimpleNamespace(session_factory=self.factory),
        ))
        dependency = get_db(request)
        session = await anext(dependency)
        session.add(SystemSetting(key="uncommitted", value="uncommitted"))
        await session.flush()
        await dependency.aclose()
        async with self.factory() as check:
            self.assertIsNone(await check.scalar(select(SystemSetting)))

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
        from app.services.setting_service import get_source_limits

        app = FastAPI()
        app.state.session_factory = self.factory
        @app.get("/limits")
        async def limits(db: AsyncSession = Depends(get_db)):
            return await get_source_limits(db)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.get("/limits")
            self.assertEqual(response.status_code, 200)
            self.assertIn("arxiv", response.json())

    async def test_lifespan_releases_database_even_when_startup_fails(self):
        from app import main

        engine_spy = Mock(wraps=self.engine)
        engine_spy.dispose = AsyncMock(wraps=self.engine.dispose)
        server = Mock(start=AsyncMock(side_effect=RuntimeError("grpc unavailable")), stop=AsyncMock())
        with patch.object(main, "create_database", return_value=(engine_spy, self.factory)), patch.object(
            main, "InternalGrpcServer", return_value=server,
        ), patch.object(main, "close_index_resources", new_callable=AsyncMock) as close_index, patch.object(
            main.AsyncPostgresSaver, "from_conn_string", return_value=AsyncMock(),
        ), patch.object(main.AsyncPostgresStore, "from_conn_string", return_value=AsyncMock()):
            with self.assertRaisesRegex(RuntimeError, "grpc unavailable"):
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
        app = FastAPI()
        with patch.object(main, "create_database", return_value=(engine_spy, self.factory)), patch.object(
            main, "InternalGrpcServer", return_value=server,
        ), patch.object(
            main, "nacos_registry", nacos,
        ), patch.object(main, "close_index_resources", new_callable=AsyncMock), patch.object(
            main, "close_task_rag_batch_runner", new_callable=AsyncMock,
        ), patch.object(main.paper_service_grpc_channel_pool, "close", new_callable=AsyncMock), patch.object(
            main.AsyncPostgresSaver, "from_conn_string", return_value=AsyncMock(),
        ), patch.object(main.AsyncPostgresStore, "from_conn_string", return_value=AsyncMock()):
            async with main.lifespan(app):
                self.assertIs(app.state.session_factory, self.factory)
                engine_spy.dispose.assert_not_awaited()
        server.stop.assert_awaited_once()
        nacos.stop.assert_awaited_once()
        engine_spy.dispose.assert_awaited_once()
