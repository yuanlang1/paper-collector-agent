import asyncio
import os
import sys

from celery import Celery, signals
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

from app.database import create_database

# 任务模块列表（Worker 启动时自动导入）
TASK_MODULES = [
    "app.tasks.modules.memory.extract_userprofile",
    "app.tasks.modules.memory.extract_fact",
    "app.tasks.modules.memory.extract_episode",
    "app.tasks.modules.memory.maintain",
    "app.tasks.modules.memory.healthcheck",
]

celery_app = Celery(
    "paper_collector_memory",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://127.0.0.1:6380/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://127.0.0.1:6380/1"),
    include=TASK_MODULES,
)

celery_app.conf.update(
    task_default_queue="memory",
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
    broker_connection_retry_on_startup=True,
    broker_connection_timeout=3,
    task_publish_retry=False,
    worker_prefetch_multiplier=1,
    timezone="Asia/Shanghai",
    beat_schedule={
        "memory-maintenance": {"task": "memory.maintain", "schedule": 60.0},
    },
)

_worker_event_loop: asyncio.AbstractEventLoop | None = None
_worker_async_engine: AsyncEngine | None = None
_worker_session_maker: async_sessionmaker[AsyncSession] | None = None


def get_worker_loop() -> asyncio.AbstractEventLoop:
    if _worker_event_loop is None:
        raise RuntimeError("Worker event loop is not initialized")
    return _worker_event_loop


def get_worker_session_maker() -> async_sessionmaker[AsyncSession]:
    if _worker_session_maker is None:
        raise RuntimeError("Worker session factory is not initialized")
    return _worker_session_maker


@signals.worker_process_init.connect
def init_worker_process(**kwargs) -> None:
    global _worker_event_loop, _worker_async_engine, _worker_session_maker
    if _worker_event_loop is not None:
        return
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        engine, factory = create_database(poolclass=NullPool)
    except BaseException:
        loop.close()
        asyncio.set_event_loop(None)
        raise
    _worker_event_loop, _worker_async_engine, _worker_session_maker = (
        loop,
        engine,
        factory,
    )


@signals.worker_shutdown.connect
@signals.worker_process_shutdown.connect
def shutdown_worker_process(**kwargs) -> None:
    global _worker_event_loop, _worker_async_engine, _worker_session_maker
    loop = _worker_event_loop
    if loop is None:
        return

    async def close_resources():
        pending = asyncio.all_tasks() - {asyncio.current_task()}
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await loop.shutdown_asyncgens()
        try:
            await _worker_async_engine.dispose()
        finally:
            await loop.shutdown_default_executor()

    try:
        loop.run_until_complete(close_resources())
    finally:
        loop.close()
        asyncio.set_event_loop(None)
        _worker_event_loop = _worker_async_engine = _worker_session_maker = None
