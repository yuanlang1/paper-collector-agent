import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.history import chat_log, stream_events
from app.history.sqlite import HistoryDatabase
from app.models.agent_history import AgentConversation, AgentMessage, AgentRun
from app.runtime.agent_runtime import AgentRuntime


class DeletionTests(unittest.IsolatedAsyncioTestCase):
    async def test_scoped_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            events = HistoryDatabase(Path(directory) / "events.db")
            events.initialize()
            engine = create_engine(f"sqlite:///{directory}/history.db")
            Base.metadata.create_all(engine, tables=[AgentConversation.__table__, AgentRun.__table__, AgentMessage.__table__])
            sessions = sessionmaker(bind=engine)
            store = chat_log.AgentHistoryStore(sessions)
            with sessions.begin() as db:
                db.add(AgentConversation(creator_id=1, conversation_id="target", thread_id="thread", status="ACTIVE"))
            await store.run(chat_log.start_turn, user_id="1", conversation_id="target", run_id="run", user_content="hello")
            for user, conversation, run in [("1", "target", "run"), ("2", "target", "other-user"), ("1", "other", "other-conversation")]:
                await events.run(stream_events.append, user_id=user, conversation_id=conversation, run_id=run, event_name="started", data={})
            runtime = AgentRuntime.__new__(AgentRuntime)
            runtime.history_db = events
            runtime.history_store = store
            runtime.agent_service = SimpleNamespace(checkpointer=SimpleNamespace(adelete_thread=AsyncMock()))
            runtime._active_stream_tasks = {}
            runtime._active_stream_conversations = {}
            with sessions() as db:
                await runtime.delete_conversation(user_id="1", conversation_id="target", db=db)
            await events.run(stream_events.delete_conversation, user_id="1", conversation_id="target")
            for user, run, expected in [("1", "run", 0), ("2", "other-user", 1), ("1", "other-conversation", 1)]:
                rows = await events.run(stream_events.list_after, user_id=user, run_id=run, after_sequence=None, limit=100)
                self.assertEqual(len(rows), expected)
            with sessions() as db:
                for model in (AgentConversation, AgentRun, AgentMessage):
                    self.assertIsNone(db.scalar(select(model)))
            engine.dispose()

    async def test_sqlite_transaction_rolls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            database = HistoryDatabase(Path(directory) / "events.db")
            database.initialize()
            stream_events.append(database, user_id="1", conversation_id="c", run_id="r", event_name="started", data={})
            with self.assertRaises(RuntimeError):
                with database.transaction() as connection:
                    connection.execute("DELETE FROM agent_stream_events")
                    raise RuntimeError("rollback")
            self.assertIsNotNone(stream_events.latest(database, user_id="1", run_id="r"))
