import asyncio
import subprocess
import sys
import unittest
from contextlib import asynccontextmanager
from contextvars import ContextVar
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from celery import signals
from celery.concurrency.solo import TaskPool
from sqlalchemy.pool import NullPool

from app.tasks.celery_utils import run_async_task
from app.tasks.modules.memory import (
    extract_episode as episode_tasks,
    extract_fact as fact_tasks,
    extract_userprofile as profile_tasks,
    maintain as maintenance_tasks,
)

runtime = import_module("app.tasks.celery_app")


class TaskDiscoveryTests(unittest.TestCase):
    def test_worker_loads_one_task_per_module_without_business_imports(self):
        script = """
import sys
from importlib import import_module

runtime = import_module("app.tasks.celery_app")
assert not any(name.startswith("app.tasks.modules.memory") for name in sys.modules)
runtime.celery_app.loader.import_default_modules()
runtime.celery_app.finalize()
expected = {
    "memory.extract_userprofile": "extract_userprofile",
    "memory.extract_fact": "extract_fact",
    "memory.extract_episode": "extract_episode",
    "memory.maintain": "maintain",
    "memory.healthcheck": "healthcheck",
}
assert set(runtime.TASK_MODULES) == {
    "app.tasks.modules.memory." + module for module in expected.values()
}
registered = {name: task for name, task in runtime.celery_app.tasks.items() if name.startswith("memory.")}
assert set(registered) == set(expected), registered
for name, module in expected.items():
    module_path = "app.tasks.modules.memory." + module
    assert registered[name].__module__ == module_path
    assert sum(task.__module__ == module_path for task in registered.values()) == 1
assert "app.services.agent_run_service" not in sys.modules
assert runtime._worker_event_loop is None
assert runtime._worker_async_engine is None
assert runtime._worker_session_maker is None
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class CeleryRuntimeTests(unittest.TestCase):
    def setUp(self):
        policy = asyncio.get_event_loop_policy()
        self.addCleanup(asyncio.set_event_loop_policy, policy)
        self.engine = SimpleNamespace(dispose=AsyncMock())
        self.sessions = []
        self.factory = Mock(side_effect=self.open_session)
        self.create = self.enterContext(patch.object(
            runtime, "create_database", return_value=(self.engine, self.factory),
        ))
        self.addCleanup(runtime.shutdown_worker_process)
        runtime.init_worker_process()

    @asynccontextmanager
    async def open_session(self):
        session = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock(), closed=False)
        self.sessions.append(session)
        try:
            yield session
        finally:
            session.closed = True

    def test_solo_initialization_reuses_resources_without_opening_sessions(self):
        loop = runtime.get_worker_loop()
        TaskPool(limit=1)
        self.assertIs(runtime.get_worker_loop(), loop)
        self.assertIs(runtime.get_worker_session_maker(), self.factory)
        self.create.assert_called_once_with(poolclass=NullPool)
        self.factory.assert_not_called()

    def test_consecutive_tasks_share_loop_but_isolate_sessions_and_context(self):
        observed = []
        marker = ContextVar("task_marker", default=None)

        async def process(session):
            self.assertIsNone(marker.get())
            marker.set("current task")
            observed.append(asyncio.get_running_loop())
            return "saved"

        for number in range(2):
            self.assertEqual(run_async_task(str(number), "extract", process), "saved")
        self.assertEqual(observed, [runtime.get_worker_loop()] * 2)
        self.assertIsNot(*self.sessions)
        for session in self.sessions:
            session.commit.assert_awaited_once()
            session.rollback.assert_not_awaited()
            self.assertTrue(session.closed)

    def test_failure_rolls_back_and_the_next_task_still_runs(self):
        async def fail(session):
            raise ValueError("extraction failed")

        with self.assertLogs("app.tasks.celery_utils", level="ERROR"), self.assertRaisesRegex(ValueError, "extraction failed"):
            run_async_task("failed", "extract", fail)
        self.sessions[0].rollback.assert_awaited_once()
        self.sessions[0].commit.assert_not_awaited()
        self.assertTrue(self.sessions[0].closed)
        self.assertEqual(run_async_task("next", "extract", AsyncMock(return_value="ok")), "ok")

    def test_commit_failure_rolls_back_and_closes_the_session(self):
        async def process(session):
            session.commit.side_effect = RuntimeError("commit failed")

        with self.assertLogs("app.tasks.celery_utils", level="ERROR"), self.assertRaisesRegex(RuntimeError, "commit failed"):
            run_async_task("failed", "extract", process)
        self.sessions[0].rollback.assert_awaited_once()
        self.assertTrue(self.sessions[0].closed)

    def test_each_task_passes_celery_id_and_its_own_process_to_the_wrapper(self):
        for task_module, task, name in (
            (profile_tasks, profile_tasks.extract_userprofile, "_extract_userprofile"),
            (fact_tasks, fact_tasks.extract_fact, "_extract_fact"),
            (episode_tasks, episode_tasks.extract_episode, "_extract_episode"),
            (maintenance_tasks, maintenance_tasks.maintain, "maintain_jobs"),
        ):
            with self.subTest(task=task.name), patch.object(task_module, name, new=AsyncMock(return_value="ok")) as process:
                task.push_request(id="celery-task-id")
                try:
                    with self.assertLogs("app.tasks.celery_utils", level="INFO") as logs:
                        self.assertEqual(task.run() if task is maintenance_tasks.maintain else task.run("job-id"), "ok")
                finally:
                    task.pop_request()
                args = () if task is maintenance_tasks.maintain else ("job-id",)
                process.assert_awaited_once_with(self.sessions[-1], *args)
                self.assertTrue(any(f"{task.name}_started task_id=celery-task-id" in entry for entry in logs.output))

    def test_shutdown_signals_cancel_pending_work_and_dispose_once(self):
        loop = runtime.get_worker_loop()
        closed = []

        async def pending():
            try:
                await asyncio.Event().wait()
            finally:
                closed.append(True)

        loop.create_task(pending())
        loop.run_until_complete(asyncio.sleep(0))
        signals.worker_shutdown.send(sender=None)
        signals.worker_process_shutdown.send(sender=None, pid=0, exitcode=0)
        self.assertEqual(closed, [True])
        self.assertTrue(loop.is_closed())
        self.engine.dispose.assert_awaited_once()
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            runtime.get_worker_loop()
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            runtime.get_worker_session_maker()

    def test_failed_initialization_closes_loop_without_publishing_resources(self):
        runtime.shutdown_worker_process()
        loop = asyncio.new_event_loop()
        with patch.object(runtime.asyncio, "new_event_loop", return_value=loop), patch.object(
            runtime, "create_database", side_effect=RuntimeError("invalid database settings"),
        ), self.assertRaisesRegex(RuntimeError, "invalid database settings"):
            runtime.init_worker_process()
        self.assertTrue(loop.is_closed())
        self.assertIsNone(runtime._worker_async_engine)
        self.assertIsNone(runtime._worker_session_maker)


if __name__ == "__main__":
    unittest.main()
