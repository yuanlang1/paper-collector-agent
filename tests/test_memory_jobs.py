import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, func, select, update

from app.db.models import AgentRun, AgentThread, MemoryJob, MemoryJobKind, MemoryJobStatus
from app.db.repositories import memory_job
from tests.test_postgres_models import PostgresDatabaseTestCase


class MemoryJobTests(PostgresDatabaseTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        async with self.factory.begin() as db:
            thread = AgentThread(user_id=11, thread_id="thread")
            db.add(thread)
            await db.flush()
            db.add(AgentRun(
                user_id=11, thread_id="thread", run_id="run", status="completed",
                run_metadata={}, finished_at=datetime.now(UTC),
            ))
            await db.flush()
            self.thread_update_time = thread.modification_time

    async def create_run_jobs(self, db):
        return await memory_job.create_run_jobs(
            db, user_id=11, thread_id="thread", run_id="run",
            checkpoint_id="checkpoint", end_id="answer",
        )

    async def create_episode(self, db, *, version=None, end_id="answer"):
        return await memory_job.create_episode_job(
            db, user_id=11, thread_id="thread", run_id="run",
            checkpoint_id="checkpoint", end_id=end_id,
            source_thread_update_time=version or self.thread_update_time,
        )

    async def expire(self, db, job_id):
        await db.execute(update(MemoryJob).where(MemoryJob.job_id == job_id).values(
            lease_expires_at=func.clock_timestamp() - timedelta(seconds=1),
        ))

    async def test_run_jobs_are_idempotent_and_do_not_commit(self):
        self.assertNotIn("model_id", MemoryJob.__table__.c)
        self.assertNotIn("start_id", MemoryJob.__table__.c)
        async with self.factory() as db:
            jobs = await self.create_run_jobs(db)
            self.assertTrue(db.in_transaction())
            self.assertEqual({j.kind for j in jobs}, {"userprofile", "fact"})
            again = await memory_job.create_run_jobs(
                db, user_id=11, thread_id="thread", run_id="run",
                checkpoint_id="later-checkpoint", end_id="later-answer",
            )
            self.assertEqual([j.job_id for j in jobs], [j.job_id for j in again])
            self.assertEqual({j.end_id for j in again}, {"answer"})
            await db.rollback()
        async with self.factory() as db:
            self.assertEqual(await db.scalar(select(func.count()).select_from(MemoryJob)), 0)

    async def test_creation_requires_an_owned_completed_run(self):
        async with self.factory.begin() as db:
            for fields in (
                dict(user_id=22, thread_id="thread", run_id="run"),
                dict(user_id=11, thread_id="another-thread", run_id="run"),
            ):
                with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, "Completed run"):
                    await memory_job.create_run_jobs(
                        db, **fields, checkpoint_id="checkpoint", end_id="answer",
                    )
            await db.execute(update(AgentRun).values(status="running"))
            with self.assertRaisesRegex(ValueError, "Completed run"):
                await self.create_run_jobs(db)

    async def test_only_one_worker_can_claim_and_finish_a_job(self):
        async with self.factory.begin() as db:
            job_id = (await self.create_run_jobs(db))[0].job_id

        async def claim():
            async with self.factory.begin() as db:
                return await memory_job.claim(db, job_id=job_id, lease_seconds=60)

        claims = await asyncio.gather(claim(), claim())
        self.assertEqual(sum(job is not None for job in claims), 1)
        claimed = next(job for job in claims if job is not None)
        self.assertEqual(claimed.attempts, 1)
        async with self.factory.begin() as db:
            self.assertIsNone(await memory_job.complete(db, job_id=job_id, claim_token="wrong"))
            renewed = await memory_job.renew_lease(
                db, job_id=job_id, claim_token=claimed.claim_token, lease_seconds=120,
            )
            self.assertGreater(renewed.lease_expires_at, claimed.lease_expires_at)
            completed = await memory_job.complete(db, job_id=job_id, claim_token=claimed.claim_token)
            self.assertEqual(completed.status, MemoryJobStatus.COMPLETED)
            self.assertIsNotNone(completed.completed_at)
            self.assertIsNone(completed.claim_token)
            self.assertIsNone(await memory_job.complete(db, job_id=job_id, claim_token=claimed.claim_token))
            self.assertIsNone(await memory_job.claim(db, job_id=job_id, lease_seconds=60))

    async def test_retry_delay_and_attempt_limit(self):
        async with self.factory.begin() as db:
            job_id = (await self.create_run_jobs(db))[0].job_id
            claimed = await memory_job.claim(db, job_id=job_id, lease_seconds=60)
            retried = await memory_job.schedule_retry(
                db, job_id=job_id, claim_token=claimed.claim_token,
                available_at=datetime.now(UTC) + timedelta(hours=1),
                error_message="temporary failure", max_attempts=2,
            )
            self.assertEqual(retried.status, MemoryJobStatus.PENDING)
            self.assertIsNone(await memory_job.claim(db, job_id=job_id, lease_seconds=60))
            self.assertNotIn(job_id, [j.job_id for j in await memory_job.list_dispatchable(db)])
            await db.execute(update(MemoryJob).where(MemoryJob.job_id == job_id).values(
                available_at=func.clock_timestamp(),
            ))
            claimed = await memory_job.claim(db, job_id=job_id, lease_seconds=60)
            failed = await memory_job.schedule_retry(
                db, job_id=job_id, claim_token=claimed.claim_token,
                available_at=datetime.now(UTC), error_message="again", max_attempts=2,
            )
            self.assertEqual((failed.status, failed.attempts), (MemoryJobStatus.FAILED, 2))
            self.assertIsNone(await memory_job.claim(db, job_id=job_id, lease_seconds=60))

    async def test_expired_worker_is_fenced_and_recovery_has_an_attempt_limit(self):
        async with self.factory.begin() as db:
            job_id = (await self.create_run_jobs(db))[0].job_id
            first = await memory_job.claim(db, job_id=job_id, lease_seconds=60)
            old_token = first.claim_token
            await self.expire(db, job_id)
            self.assertIsNone(await memory_job.complete(db, job_id=job_id, claim_token=old_token))
            self.assertIsNone(await memory_job.renew_lease(
                db, job_id=job_id, claim_token=old_token, lease_seconds=60,
            ))
            recovered = await memory_job.recover_expired_jobs(db, max_attempts=2)
            self.assertEqual([j.job_id for j in recovered], [job_id])
            second = await memory_job.claim(db, job_id=job_id, lease_seconds=60)
            self.assertNotEqual(second.claim_token, old_token)
            self.assertIsNone(await memory_job.complete(db, job_id=job_id, claim_token=old_token))
            await self.expire(db, job_id)
            failed = (await memory_job.recover_expired_jobs(db, max_attempts=2))[0]
            self.assertEqual(failed.status, MemoryJobStatus.FAILED)

    async def test_episode_deduplication_cursor_and_thread_time(self):
        async with self.factory.begin() as db:
            first = await self.create_episode(db)
            self.assertEqual(first.kind, MemoryJobKind.EPISODE)
            self.assertIsNone(await self.create_episode(db))
            later = self.thread_update_time + timedelta(minutes=1)
            self.assertIsNone(await self.create_episode(db, version=later))
            claimed = await memory_job.claim(db, job_id=first.job_id, lease_seconds=60)
            await memory_job.mark_skipped(db, job_id=first.job_id, claim_token=claimed.claim_token)
            self.assertIsNone(await self.create_episode(db))
            self.assertIsNone(await memory_job.get_last_completed_episode(db, user_id=11, thread_id="thread"))

            second = await self.create_episode(db, version=later, end_id="second-answer")
            claimed = await memory_job.claim(db, job_id=second.job_id, lease_seconds=60)
            await memory_job.complete(db, job_id=second.job_id, claim_token=claimed.claim_token)
            last = await memory_job.get_last_completed_episode(db, user_id=11, thread_id="thread")
            self.assertEqual(last.end_id, "second-answer")
            self.assertIsNone(await memory_job.get_last_completed_episode(db, user_id=22, thread_id="thread"))

            third = await self.create_episode(db, version=later + timedelta(minutes=1))
            claimed = await memory_job.claim(db, job_id=third.job_id, lease_seconds=60)
            await memory_job.mark_failed(
                db, job_id=third.job_id, claim_token=claimed.claim_token, error_message="bad source",
            )
            last = await memory_job.get_last_completed_episode(db, user_id=11, thread_id="thread")
            self.assertEqual(last.job_id, second.job_id)
            self.assertEqual(await db.scalar(select(AgentThread.modification_time)), self.thread_update_time)

    async def test_run_deletion_cascades_to_jobs(self):
        async with self.factory.begin() as db:
            jobs = await self.create_run_jobs(db)
            await self.create_episode(db)
            self.assertIsNotNone(await memory_job.get_by_job_id(db, job_id=jobs[0].job_id))
            await db.execute(delete(AgentRun).where(AgentRun.run_id == "run"))
            self.assertEqual(await db.scalar(select(func.count()).select_from(MemoryJob)), 0)
