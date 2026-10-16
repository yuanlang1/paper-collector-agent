"""Opt-in MySQL integration tests using a temporary, isolated database.

Run with AGENT_MYSQL_TESTS=1. The configured MySQL user needs CREATE/DROP
DATABASE permissions. No tables in the configured business database are changed.
"""
import asyncio
import os
import unittest
from uuid import uuid4

from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models import AgentMessage, AgentSession, AgentSummary
from app.db.repositories import agent_message, agent_session, agent_summary
from app.db.session import create_database


@unittest.skipUnless(os.environ.get("AGENT_MYSQL_TESTS") == "1", "requires MySQL opt-in")
class AgentRepositoriesMySQLTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.admin, _ = create_database()
        self.database = "agent_repo_test_" + uuid4().hex
        self.engine = None
        self.created = False
        self.addAsyncCleanup(self.cleanup_database)
        async with self.admin.connect() as conn:
            await conn.execute(text(f"CREATE DATABASE `{self.database}` CHARACTER SET utf8mb4"))
            self.created = True
        self.engine = create_async_engine(self.admin.url.set(database=self.database))
        self.factory = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as conn:
            for _ in range(2):
                for model in (AgentSession, AgentMessage, AgentSummary):
                    await conn.run_sync(model.__table__.create, checkfirst=True)

    async def cleanup_database(self):
        if self.engine is not None:
            await self.engine.dispose()
        try:
            if self.created:
                # This identifier is generated above, never supplied by settings or a caller.
                async with self.admin.connect() as conn:
                    await conn.execute(text(f"DROP DATABASE `{self.database}`"))
        finally:
            await self.admin.dispose()

    async def test_messages_json_pagination_and_case_sensitive_identifiers(self):
        async with self.factory.begin() as db:
            first = await agent_session.create(
                db,
                user_id=1,
                thread_id="Thread",
            )
            second = await agent_session.create(
                db,
                user_id=1,
                thread_id="thread",
            )
            self.assertGreater(second.id, first.id)
            self.assertEqual(first.creator_id, 1)
            self.assertIsNotNone(first.creation_time)
            rows = await agent_message.add_many(
                db,
                user_id=1,
                thread_id="Thread",
                messages=[
                    {"run_id": "Run", "role": "user", "content": "你好"},
                    {
                        "run_id": "Run",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "回答"}],
                        "metadata": {"message_id": "model-1"},
                    },
                    {"run_id": "run", "role": "user", "content": "next"},
                ],
            )
            ids = [row.id for row in rows]
        async with self.factory() as db:
            page = await agent_message.list_by_thread(
                db,
                user_id=1,
                thread_id="Thread",
                limit=2,
            )
            self.assertEqual([row.id for row in page], ids[1:])
            self.assertEqual(page[0].content, [{"type": "text", "text": "回答"}])
            self.assertEqual(page[0].message_metadata, {"message_id": "model-1"})
            older = await agent_message.list_by_thread(
                db,
                user_id=1,
                thread_id="Thread",
                before_id=ids[1],
            )
            self.assertEqual([row.id for row in older], ids[:1])
            self.assertEqual(older[0].content, "你好")
            self.assertEqual(older[0].message_metadata, {})
            self.assertEqual(
                len(
                    await agent_message.list_by_run(
                        db,
                        user_id=1,
                        thread_id="Thread",
                        run_id="Run",
                    )
                ),
                2,
            )
            self.assertEqual(
                len(
                    await agent_message.list_by_range(
                        db,
                        user_id=1,
                        thread_id="Thread",
                        start_id=ids[0],
                        end_id=ids[1],
                    )
                ),
                2,
            )

    async def test_summary_upsert_and_access_checks(self):
        async with self.factory.begin() as db:
            await agent_session.create(
                db,
                user_id=1,
                thread_id="one",
            )
            await agent_session.create(
                db,
                user_id=2,
                thread_id="two",
            )
            first = await agent_message.add(
                db,
                user_id=1,
                thread_id="one",
                run_id="r",
                role="user",
                content="a",
            )
            last = await agent_message.add(
                db,
                user_id=1,
                thread_id="one",
                run_id="r",
                role="assistant",
                content="b",
            )
            other = await agent_message.add(
                db,
                user_id=2,
                thread_id="two",
                run_id="r",
                role="user",
                content="c",
            )
            summary = await agent_summary.save(
                db,
                user_id=1,
                thread_id="one",
                start_id=first.id,
                end_id=last.id,
                summary="old",
            )
            saved_id = summary.id
            updated = await agent_summary.save(
                db,
                user_id=1,
                thread_id="one",
                start_id=first.id,
                end_id=last.id,
                summary="new",
            )
            self.assertEqual(updated.id, saved_id)
            self.assertEqual(updated.summary, "new")
            self.assertEqual(
                await db.scalar(select(func.count()).select_from(AgentSummary)), 1
            )
            self.assertEqual(
                (
                    await agent_summary.get_latest(
                        db,
                        user_id=1,
                        thread_id="one",
                    )
                ).id,
                saved_id,
            )
            self.assertEqual(
                (
                    await agent_summary.get_by_range(
                        db,
                        user_id=1,
                        thread_id="one",
                        start_id=first.id,
                        end_id=last.id,
                    )
                ).summary,
                "new",
            )
            for start, end in [
                (last.id, first.id),
                (first.id, other.id),
                (first.id, other.id + 100),
            ]:
                with self.assertRaises(ValueError):
                    await agent_summary.save(
                        db,
                        user_id=1,
                        thread_id="one",
                        start_id=start,
                        end_id=end,
                        summary="bad",
                    )
            with self.assertRaises(ValueError):
                await agent_message.list_by_thread(
                    db,
                    user_id=2,
                    thread_id="one",
                )
            with self.assertRaises(ValueError):
                await agent_message.add(
                    db,
                    user_id=2,
                    thread_id="one",
                    run_id="r",
                    role="user",
                    content="bad",
                )
            with self.assertRaises(ValueError):
                await agent_summary.get_latest(
                    db,
                    user_id=2,
                    thread_id="one",
                )
            with self.assertRaises(ValueError):
                await agent_summary.save(
                    db,
                    user_id=2,
                    thread_id="one",
                    start_id=first.id,
                    end_id=last.id,
                    summary="bad",
                )
            with self.assertRaises(ValueError):
                await agent_session.update_topic(
                    db,
                    user_id=2,
                    thread_id="one",
                    topic="bad",
                )

    async def test_constraints_and_transaction_rollback(self):
        async with self.factory.begin() as db:
            await agent_session.create(
                db,
                user_id=1,
                thread_id="one",
            )
        with self.assertRaises(IntegrityError):
            async with self.factory.begin() as db:
                await agent_session.create(
                    db,
                    user_id=1,
                    thread_id="one",
                )
        with self.assertRaises(IntegrityError):
            async with self.factory.begin() as db:
                db.add(
                    AgentMessage(
                        thread_id="missing",
                        run_id="r",
                        role="user",
                        content="bad",
                    )
                )
                await db.flush()
        with self.assertRaises(RuntimeError):
            async with self.factory.begin() as db:
                await agent_session.create(
                    db,
                    user_id=1,
                    thread_id="rollback",
                )
                await agent_message.add(
                    db,
                    user_id=1,
                    thread_id="rollback",
                    run_id="r",
                    role="user",
                    content="a",
                )
                raise RuntimeError("rollback")
        async with self.factory() as db:
            self.assertIsNone(
                await agent_session.get_by_thread_id(
                    db,
                    user_id=1,
                    thread_id="rollback",
                )
            )
            self.assertEqual(
                await db.scalar(select(func.count()).select_from(AgentMessage)), 0
            )

    async def test_modification_time_orm_and_direct_sql(self):
        async with self.factory.begin() as db:
            row = await agent_session.create(
                db,
                user_id=1,
                thread_id="one",
            )
            original = row.modification_time
        await asyncio.sleep(1.1)
        async with self.factory.begin() as db:
            row = await agent_session.update_topic(
                db,
                user_id=1,
                thread_id="one",
                topic="ORM",
            )
            await db.refresh(row)
            self.assertGreater(row.modification_time, original)
            changed = row.modification_time
            self.assertEqual(len(await agent_session.list_by_user(db, user_id=1)), 1)
            self.assertEqual(await agent_session.list_by_user(db, user_id=2), [])
        await asyncio.sleep(1.1)
        async with self.engine.begin() as conn:
            await conn.execute(
                text("UPDATE agent_session SET topic='SQL' WHERE thread_id='one'")
            )
            modified = await conn.scalar(
                text("SELECT modification_time FROM agent_session WHERE thread_id='one'")
            )
            self.assertGreater(modified, changed)


if __name__ == "__main__":
    unittest.main()
