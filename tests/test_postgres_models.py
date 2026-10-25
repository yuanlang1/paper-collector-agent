import asyncio
import os
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

from sqlalchemy import BigInteger, inspect
from sqlalchemy.engine import make_url
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateSchema, CreateTable, DropSchema
from sqlalchemy.pool import NullPool

from app.config import settings
from app.database import Base, create_database
from app.db.models import (
    AgentThread, AgentRun, LlmSetting, MemoryJob, SystemSetting,
)
from app.db.models.entity_base import EntityBase


MODELS = (
    AgentThread, SystemSetting, LlmSetting, AgentRun, MemoryJob,
)
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_POSTGRES_URL")

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class PostgresDatabaseTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if not TEST_DATABASE_URL:
            self.skipTest("set TEST_DATABASE_POSTGRES_URL to an independent PostgreSQL test database")
        schema = self.schema = "test_" + uuid4().hex
        url = make_url(TEST_DATABASE_URL)
        options = url.query.get("options", "") + f" -c search_path={schema}"
        url = url.update_query_dict({"options": options.strip()})
        with patch.object(settings, "DATABASE_POSTGRES_URL", url.render_as_string(hide_password=False)), patch.object(
            settings, "SQL_ECHO", False,
        ):
            self.engine, self.factory = create_database()
        self.addAsyncCleanup(self.engine.dispose)
        async with self.engine.begin() as connection:
            await connection.execute(CreateSchema(schema))
        self.addAsyncCleanup(self.drop_schema, schema)
        # text() ignores schema_translate_map; search_path also isolates native SQL.
        self.engine.update_execution_options(schema_translate_map={None: schema})
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def drop_schema(self, schema):
        async with self.engine.begin() as connection:
            await connection.execute(DropSchema(schema, cascade=True))


class PostgresModelTests(unittest.TestCase):
    def test_run_metadata_column_does_not_override_sqlalchemy_metadata(self):
        self.assertIn("metadata", AgentRun.__table__.c)
        self.assertNotIn("input", AgentRun.__table__.c)
        self.assertEqual(inspect(AgentRun).attrs.run_metadata.columns[0].name, "metadata")
        self.assertIs(AgentRun.metadata, Base.metadata)

    def test_all_tables_share_entity_fields_and_compile_for_postgres(self):
        dialect = postgresql.dialect()
        self.assertEqual(len(Base.metadata.tables), 5)
        for model in MODELS:
            with self.subTest(table=model.__tablename__):
                self.assertTrue(issubclass(model, EntityBase))
                table = model.__table__
                self.assertEqual(list(table.primary_key.columns.keys()), ["id"])
                self.assertIsInstance(table.c.id.type, BigInteger)
                for name in ("creation_time", "modification_time"):
                    self.assertTrue(table.c[name].type.timezone)
                    self.assertFalse(table.c[name].nullable)
                self.assertIsNotNone(table.c.modification_time.onupdate)
                self.assertNotIn("created_at", table.c)
                self.assertNotIn("updated_at", table.c)
                ddl = str(CreateTable(table).compile(dialect=dialect))
                self.assertIn("BIGSERIAL", ddl)
                self.assertNotIn("ON UPDATE", ddl)
                self.assertNotIn("utf8mb4", ddl)
                for index in table.indexes:
                    str(CreateIndex(index).compile(dialect=dialect))
        owner = LlmSetting.__table__.c.user_id
        self.assertIsInstance(owner.type, BigInteger)
        self.assertFalse(owner.nullable)
        self.assertTrue(owner.index)
        for model in (AgentThread, AgentRun):
            self.assertIsInstance(model.__table__.c.user_id.type, BigInteger)
            self.assertFalse(model.__table__.c.user_id.nullable)

    def test_config_representations_do_not_include_secrets(self):
        self.assertNotIn("test-secret", repr(SystemSetting(key="key", value="test-secret")))
        self.assertNotIn("test-secret", repr(LlmSetting(api_key="test-secret")))


class PostgresDriverTests(unittest.IsolatedAsyncioTestCase):
    async def test_postgres_url_selects_async_psycopg_without_connecting(self):
        with patch.object(settings, "DATABASE_POSTGRES_URL", "postgresql://user:password@localhost/example"):
            engine, factory = create_database()
        try:
            self.assertEqual(engine.url.drivername, "postgresql+psycopg")
            self.assertTrue(engine.dialect.is_async)
            self.assertFalse(factory.kw["expire_on_commit"])
        finally:
            await engine.dispose()

    async def test_database_aliases_share_the_async_factory(self):
        from app.core.database import create_database as core_factory
        from app.db.session import create_database as db_factory

        self.assertIs(core_factory, create_database)
        self.assertIs(db_factory, create_database)

        with patch.object(settings, "DATABASE_POSTGRES_URL", "postgresql://user:password@localhost/example"):
            engine, _ = core_factory(poolclass=NullPool)
        try:
            self.assertEqual(engine.url.drivername, "postgresql+psycopg")
            self.assertTrue(engine.dialect.is_async)
            self.assertIsInstance(engine.pool, NullPool)
        finally:
            await engine.dispose()
