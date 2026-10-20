import asyncio
import os
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

from sqlalchemy import BigInteger, UniqueConstraint, inspect
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateSchema, CreateTable, DropSchema

from app.config import settings
from app.database import Base, create_database
from app.db.models import (
    AgentThread, AgentRun, LlmSetting,
    MemoryConsolidationCursor, MemoryEpisode, MemoryFact, SystemSetting,
)
from app.db.models.entity_base import EntityBase


MODELS = (
    AgentThread, SystemSetting,
    MemoryFact, MemoryEpisode, MemoryConsolidationCursor, LlmSetting, AgentRun,
)
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_POSTGRES_URL")

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class PostgresDatabaseTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if not TEST_DATABASE_URL:
            self.skipTest("set TEST_DATABASE_POSTGRES_URL to an independent PostgreSQL test database")
        with patch.object(settings, "DATABASE_POSTGRES_URL", TEST_DATABASE_URL), patch.object(
            settings, "SQL_ECHO", False,
        ):
            self.engine, self.factory = create_database()
        self.addAsyncCleanup(self.engine.dispose)
        schema = self.schema = "test_" + uuid4().hex
        async with self.engine.begin() as connection:
            await connection.execute(CreateSchema(schema))
        self.addAsyncCleanup(self.drop_schema, schema)
        # Every connection uses only the generated test schema, never public tables.
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
        self.assertEqual(len(Base.metadata.tables), 7)
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
        cursor = MemoryConsolidationCursor.__table__
        self.assertTrue(any(
            isinstance(c, UniqueConstraint)
            and list(c.columns.keys()) == ["user_id", "conversation_id"]
            for c in cursor.constraints
        ))
        self.assertIsInstance(MemoryEpisode.__table__.c.source_assistant_message_id.type, BigInteger)
        self.assertIsInstance(cursor.c.last_assistant_message_id.type, BigInteger)
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

    async def test_sync_factory_uses_same_postgres_url_without_connecting(self):
        from app.core.database import create_database as create_sync_database

        with patch.object(settings, "DATABASE_POSTGRES_URL", "postgresql://user:password@localhost/example"):
            engine, _ = create_sync_database()
        try:
            self.assertEqual(engine.url.drivername, "postgresql+psycopg")
            self.assertFalse(engine.dialect.is_async)
        finally:
            engine.dispose()
