import asyncio
import unittest
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.store.memory import InMemoryStore
from sqlalchemy import text

from app.common.exceptions import NotFoundException
from app.config import settings
from app.db.models import RunStatus
from app.db.repositories import agent_run, agent_thread, memory_job
from app.memory import manager as memory_manager
from app.memory.namespace import profile_namespace
from app.memory.schemas import UserProfile
from app.services import agent_run_service
from app.tasks.modules.memory import (
    common,
    extract_episode as episode_tasks,
    extract_fact as fact_tasks,
    extract_userprofile as profile_tasks,
    maintain as maintenance_tasks,
)
from tests.test_postgres_models import PostgresDatabaseTestCase


@asynccontextmanager
async def opened(value):
    yield value


class MemoryInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_checkpoint_run_and_episode_boundaries(self):
        messages = [
            HumanMessage(id="old-user", content="old"),
            AIMessage(id="old-end", content="old answer"),
            HumanMessage(id="user", content="current"),
            AIMessage(id="call", content="", tool_calls=[{
                "name": "search", "args": {}, "id": "tool-call",
            }]),
            ToolMessage(id="tool", content="result", tool_call_id="tool-call"),
            AIMessage(id="end", content="current answer"),
            HumanMessage(id="later", content="must not be included"),
        ]
        saver = SimpleNamespace(aget_tuple=AsyncMock(return_value=SimpleNamespace(
            checkpoint={"channel_values": {"messages": messages}},
        )))
        job = SimpleNamespace(thread_id="thread", checkpoint_id="fixed", end_id="end", kind="fact")
        self.assertEqual(await common.load_messages(saver, job), messages[2:6])
        saver.aget_tuple.assert_awaited_once_with({"configurable": {
            "thread_id": "thread", "checkpoint_ns": "", "checkpoint_id": "fixed",
        }})
        job.kind = "userprofile"
        self.assertEqual(await common.load_messages(saver, job), messages[2:6])
        job.kind = "episode"
        self.assertEqual(await common.load_messages(saver, job), messages[:6])
        self.assertEqual(await common.load_messages(saver, job, after_id="old-end"), messages[2:6])
        self.assertEqual(await common.load_messages(saver, job, after_id="end"), [])
        with self.assertRaisesRegex(ValueError, "Previous episode"):
            await common.load_messages(saver, job, after_id="missing")
        job.end_id = "missing"
        with self.assertRaisesRegex(ValueError, "end message"):
            await common.load_messages(saver, job)
        saver.aget_tuple.return_value = None
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            await common.load_messages(saver, job)

    async def test_profile_has_a_default_for_new_users_and_preserves_existing_values(self):
        store = InMemoryStore()
        for existing in (None, UserProfile(research_interests=["graphs"])):
            if existing is not None:
                await store.aput(profile_namespace("11"), "default", {
                    "kind": "UserProfile", "content": existing.model_dump(),
                })
            with patch.object(memory_manager, "create_memory_store_manager") as create:
                await memory_manager.create_profile_manager(model=Mock(), store=store, user_id="11")
            self.assertEqual(create.call_args.kwargs["default"], existing or UserProfile())
            self.assertFalse(create.call_args.kwargs["enable_inserts"])
            self.assertFalse(create.call_args.kwargs["enable_deletes"])


class MemoryTaskTests(PostgresDatabaseTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        async with self.factory.begin() as db:
            await agent_thread.create(db, user_id=11, thread_id="thread")
            await agent_run.create(db, user_id=11, thread_id="thread", run_id="run", metadata={})
            await agent_run.update_status(
                db, user_id=11, thread_id="thread", run_id="run",
                status=RunStatus.COMPLETED, expected_status=(RunStatus.PENDING,),
            )
            self.jobs = await memory_job.create_run_jobs(
                db, user_id=11, thread_id="thread", run_id="run",
                checkpoint_id="fixed", end_id="end",
            )
        self.enterContext(patch.object(common, "get_worker_session_maker", return_value=self.factory))

    async def execute(self, operation, *args):
        async with self.factory() as db:
            return await operation(db, *args)

    async def idle(self, seconds=601):
        async with self.factory.begin() as db:
            await db.execute(text("""
                UPDATE agent_thread
                SET modification_time = clock_timestamp() - :seconds * interval '1 second'
                WHERE thread_id = 'thread'
            """), {"seconds": seconds})

    async def scan(self):
        async with self.factory.begin() as db:
            return await maintenance_tasks.scan_episodes(db)

    async def episode(self):
        async with self.factory() as db:
            rows = await memory_job.list_dispatchable(db)
        return next(job for job in rows if job.kind == "episode")

    async def test_completion_and_jobs_roll_back_together_and_only_transition_once(self):
        async with self.factory() as db:
            run = await agent_run_service.create_run(db, user_id=11, thread_id="thread", metadata={})
            await agent_run_service.update_status(
                db, user_id=11, thread_id="thread", run_id=run.run_id, status=RunStatus.RUNNING,
            )
        fields = dict(user_id=11, thread_id="thread", run_id=run.run_id, checkpoint_id="new", end_id="new-end")
        async with self.factory() as db:
            with patch.object(memory_job, "create_run_jobs", side_effect=RuntimeError("insert failed")):
                with self.assertRaisesRegex(RuntimeError, "insert failed"):
                    await agent_run_service.complete_run(db, **fields)
            stored = await agent_run.get_by_run_id(db, user_id=11, thread_id="thread", run_id=run.run_id)
            self.assertEqual(stored.status, "running")
            self.assertIsNone(stored.finished_at)
        async def complete():
            async with self.factory() as db:
                return await agent_run_service.complete_run(db, **fields)
        results = await asyncio.gather(complete(), complete())
        self.assertEqual(sorted(len(jobs) for _, jobs in results), [0, 2])
        jobs = next(jobs for _, jobs in results if jobs)
        self.assertEqual({job.kind for job in jobs}, {"userprofile", "fact"})
        self.assertEqual({(job.checkpoint_id, job.end_id) for job in jobs}, {("new", "new-end")})
        async with self.factory() as db:
            self.assertEqual(len(await memory_job.list_dispatchable(db)), 4)

    async def test_dispatch_sends_only_job_ids_and_keeps_failed_publications_pending(self):
        with patch.object(profile_tasks.extract_userprofile, "delay", side_effect=ConnectionError), patch.object(
            fact_tasks.extract_fact, "delay",
        ) as fact, self.assertLogs(maintenance_tasks.logger, level="WARNING"):
            self.assertEqual((await self.execute(maintenance_tasks.maintain_jobs))["published"], 1)
        fact.assert_called_once_with(self.jobs[1].job_id)
        async with self.factory() as db:
            self.assertEqual([job.status for job in await memory_job.list_dispatchable(db)], ["pending", "pending"])

    async def test_execute_claims_once_and_finishes_after_extraction(self):
        job = self.jobs[0]
        with patch.object(common, "load_job_messages", new=AsyncMock(return_value=[])) as extract:
            result = await self.execute(profile_tasks._extract_userprofile, job.job_id)
            duplicate = await self.execute(profile_tasks._extract_userprofile, job.job_id)
        self.assertEqual((result["status"], duplicate["status"]), ("completed", "ignored"))
        extract.assert_awaited_once()
        async with self.factory() as db:
            stored = await memory_job.get_by_job_id(db, job_id=job.job_id)
        self.assertEqual(stored.attempts, 1)
        self.assertIsNone(stored.claim_token)

    async def test_wrong_task_kind_does_not_claim_a_job(self):
        result = await self.execute(fact_tasks._extract_fact, self.jobs[0].job_id)
        self.assertEqual(result["status"], "ignored")
        async with self.factory() as db:
            self.assertEqual((await memory_job.get_by_job_id(db, job_id=self.jobs[0].job_id)).attempts, 0)

    async def test_worker_wrapper_commits_jobs_on_its_own_loop_and_engine(self):
        runtime = import_module("app.tasks.celery_app")
        url = self.engine.url.render_as_string(hide_password=False)

        def execute_in_worker():
            with patch.object(settings, "DATABASE_POSTGRES_URL", url), patch.object(
                settings, "SQL_ECHO", False,
            ), patch.object(common, "get_worker_session_maker", runtime.get_worker_session_maker), patch.object(
                common, "load_job_messages", new=AsyncMock(return_value=[]),
            ):
                runtime.init_worker_process()
                loop = runtime.get_worker_loop()
                results = []
                try:
                    for job, task in zip(self.jobs, (profile_tasks.extract_userprofile, fact_tasks.extract_fact)):
                        task.push_request(id="worker-test-" + job.job_id)
                        try:
                            results.append(task.run(job.job_id))
                        finally:
                            task.pop_request()
                        self.assertIs(runtime.get_worker_loop(), loop)
                finally:
                    runtime.shutdown_worker_process()
                self.assertTrue(loop.is_closed())
                return results

        results = await asyncio.to_thread(execute_in_worker)
        self.assertEqual([result["status"] for result in results], ["completed", "completed"])
        async with self.factory() as db:
            for job in self.jobs:
                stored = await memory_job.get_by_job_id(db, job_id=job.job_id)
                self.assertEqual(stored.status, "completed")

    async def test_claim_commits_before_extraction_and_heartbeat_has_its_own_session(self):
        renewed = asyncio.Event()
        renew = memory_job.renew_lease
        async with self.factory() as job_db:
            async def load(job):
                self.assertFalse(job_db.in_transaction())
                async with self.factory() as observer:
                    stored = await memory_job.get_by_job_id(observer, job_id=job.job_id)
                    self.assertEqual(stored.status, "running")
                await asyncio.wait_for(renewed.wait(), 5)
                return []

            async def heartbeat(db, **kwargs):
                self.assertIsNot(db, job_db)
                result = await renew(db, **kwargs)
                renewed.set()
                return result

            with patch.object(common, "HEARTBEAT_SECONDS", 0.01), patch.object(
                common, "load_job_messages", side_effect=load,
            ), patch.object(memory_job, "renew_lease", side_effect=heartbeat):
                result = await profile_tasks._extract_userprofile(job_db, self.jobs[0].job_id)
        self.assertEqual(result["status"], "completed")

    async def test_transient_failure_retries_but_missing_config_fails(self):
        for job, execute, error, expected in (
            (self.jobs[0], profile_tasks._extract_userprofile, ValueError("malformed model response"), "pending"),
            (self.jobs[1], fact_tasks._extract_fact, NotFoundException("missing config"), "failed"),
        ):
            with patch.object(common, "load_job_messages", new=AsyncMock(return_value=[HumanMessage(content="input")])), patch.object(
                common, "get_model", new=AsyncMock(side_effect=error),
            ), self.assertLogs(common.logger, level="WARNING"):
                result = await self.execute(execute, job.job_id)
            self.assertEqual(result["status"], expected)
        async with self.factory() as db:
            retried = await memory_job.get_by_job_id(db, job_id=self.jobs[0].job_id)
            self.assertGreater(retried.available_at, datetime.now(UTC))
            self.assertIsNone(retried.claim_token)
            self.assertEqual(await memory_job.list_dispatchable(db), [])

    async def test_lease_loss_cancels_extraction_without_completing(self):
        started, cancelled = asyncio.Event(), asyncio.Event()
        async def work(*args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        async def lose_lease(*args, **kwargs):
            await started.wait()
            return None
        with patch.object(common, "HEARTBEAT_SECONDS", 0), patch.object(
            memory_job, "renew_lease", new=AsyncMock(side_effect=lose_lease),
        ), patch.object(common, "load_job_messages", side_effect=work):
            result = await self.execute(profile_tasks._extract_userprofile, self.jobs[0].job_id)
        self.assertEqual(result["status"], "lease_lost")
        self.assertTrue(cancelled.is_set())
        async with self.factory() as db:
            self.assertEqual((await memory_job.get_by_job_id(db, job_id=self.jobs[0].job_id)).status, "running")

    async def test_each_kind_uses_the_users_small_model_and_corresponding_manager(self):
        await self.idle()
        await self.scan()
        jobs = [*self.jobs, await self.episode()]
        saver = SimpleNamespace(aget_tuple=AsyncMock(return_value=SimpleNamespace(
            checkpoint={"channel_values": {"messages": [
                HumanMessage(id="user", content="remember this"), AIMessage(id="end", content="done"),
            ]}},
        )))
        store = InMemoryStore()
        config = SimpleNamespace(model="small", base_url="https://example.test/v1", api_key="test-key")
        for job, factory_name, execute, task_module in zip(
            jobs,
            ("create_profile_manager", "create_fact_manager", "create_episode_manager"),
            (profile_tasks._extract_userprofile, fact_tasks._extract_fact, episode_tasks._extract_episode),
            (profile_tasks, fact_tasks, episode_tasks),
        ):
            manager = SimpleNamespace(ainvoke=AsyncMock(return_value=[]))
            factory = AsyncMock(return_value=manager) if job.kind == "userprofile" else Mock(return_value=manager)
            with self.subTest(kind=job.kind), patch.object(
                common.AsyncPostgresSaver, "from_conn_string", side_effect=lambda *args: opened(saver),
            ), patch.object(task_module, "open_memory_store", side_effect=lambda: opened(store)), patch.object(
                common, "get_small_llm_config", new=AsyncMock(return_value=config),
            ) as small_model, patch.object(common, "ChatOpenAI") as model, patch.object(task_module, factory_name, factory):
                self.assertEqual((await self.execute(execute, job.job_id))["status"], "completed")
                self.assertEqual(small_model.call_args.kwargs, {"user_id": 11})
                model.assert_called_once_with(model="small", base_url=config.base_url, api_key="test-key", streaming=False)
                self.assertEqual(factory.call_args.kwargs["user_id"], "11")
                manager.ainvoke.assert_awaited_once()
                self.assertEqual([m.id for m in manager.ainvoke.call_args.args[0]["messages"]], ["user", "end"])
                if job.kind == "episode":
                    self.assertEqual(factory.call_args.kwargs["job_id"], job.job_id)

    async def test_episode_requires_idle_time_and_latest_completed_run(self):
        await self.idle(599)
        self.assertEqual(await self.scan(), 0)
        await self.idle(601)
        async with self.factory.begin() as db:
            await agent_run.create(db, user_id=11, thread_id="thread", run_id="latest", metadata={})
        for status in ("pending", "running", "failed", "cancelled", "interrupted"):
            async with self.factory.begin() as db:
                await db.execute(text("UPDATE agent_run SET status = :status WHERE run_id = 'latest'"), {"status": status})
            self.assertEqual(await self.scan(), 0, status)
        async with self.factory.begin() as db:
            await db.execute(text("DELETE FROM agent_run WHERE run_id = 'latest'"))
        self.assertEqual(await self.scan(), 1)
        job = await self.episode()
        self.assertEqual((job.run_id, job.checkpoint_id, job.end_id), ("run", "fixed", "end"))
        self.assertEqual(await self.scan(), 0)

    async def test_concurrent_scans_create_one_episode(self):
        await self.idle()
        self.assertEqual(sum(await asyncio.gather(self.scan(), self.scan())), 1)

    async def test_scan_skips_a_thread_locked_by_a_new_request(self):
        await self.idle()
        async with self.factory.begin() as db:
            await agent_thread.get_by_thread_id(db, user_id=11, thread_id="thread", for_update=True)
            self.assertEqual(await self.scan(), 0)
            await agent_thread.touch(db, user_id=11, thread_id="thread")
        self.assertEqual(await self.scan(), 0)

    async def test_recheck_after_lock_rejects_new_activity(self):
        await self.idle()
        async with self.factory.begin() as db:
            self.assertEqual(len(await memory_job.lock_idle_threads(db)), 1)
            await agent_thread.touch(db, user_id=11, thread_id="thread")
            self.assertIsNone(await memory_job.get_episode_source(db, user_id=11, thread_id="thread"))

    async def test_queued_episode_is_skipped_if_thread_has_become_active(self):
        await self.idle()
        await self.scan()
        episode = await self.episode()
        async with self.factory.begin() as db:
            await agent_thread.touch(db, user_id=11, thread_id="thread")
        with patch.object(episode_tasks, "open_memory_store") as store:
            result = await self.execute(episode_tasks._extract_episode, episode.job_id)
        self.assertEqual(result["status"], "skipped")
        store.assert_not_called()

    async def test_empty_episode_advances_cursor_and_new_messages_are_incremental(self):
        await self.idle()
        await self.scan()
        first = await self.episode()
        with patch.object(common, "load_job_messages", new=AsyncMock(return_value=[])):
            self.assertEqual((await self.execute(episode_tasks._extract_episode, first.job_id))["status"], "completed")
        await self.idle(602)
        self.assertEqual(await self.scan(), 0)
        async with self.factory() as db:
            later_run = await agent_run_service.create_run(db, user_id=11, thread_id="thread", metadata={})
            await agent_run_service.update_status(db, user_id=11, thread_id="thread", run_id=later_run.run_id, status=RunStatus.RUNNING)
            await agent_run_service.complete_run(db, user_id=11, thread_id="thread", run_id=later_run.run_id, checkpoint_id="later", end_id="later-end")
        await self.idle()
        self.assertEqual(await self.scan(), 1)
        second = await self.episode()
        with patch.object(common, "load_messages", new=AsyncMock(return_value=[])) as load, patch.object(
            common.AsyncPostgresSaver, "from_conn_string", side_effect=lambda *args: opened(Mock()),
        ), patch.object(episode_tasks, "open_memory_store") as store:
            result = await self.execute(episode_tasks._extract_episode, second.job_id)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(load.call_args.kwargs, {"after_id": "end"})
        store.assert_not_called()

    async def test_maintenance_recovers_expired_jobs_creates_episode_and_redispatches(self):
        await self.idle()
        async with self.factory.begin() as db:
            await memory_job.claim(db, job_id=self.jobs[0].job_id, lease_seconds=60)
            await db.execute(text("""
                UPDATE memory_job
                SET lease_expires_at = clock_timestamp() - interval '1 second'
                WHERE job_id = :job_id
            """), {"job_id": self.jobs[0].job_id})
        with patch.object(profile_tasks.extract_userprofile, "delay") as profile, patch.object(
            fact_tasks.extract_fact, "delay",
        ) as fact, patch.object(episode_tasks.extract_episode, "delay") as episode:
            result = await self.execute(maintenance_tasks.maintain_jobs)
        self.assertEqual(result, {"recovered": 1, "episodes": 1, "published": 3})
        profile.assert_called_once_with(self.jobs[0].job_id)
        fact.assert_called_once_with(self.jobs[1].job_id)
        episode.assert_called_once_with((await self.episode()).job_id)
