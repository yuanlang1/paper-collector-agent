import logging
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.tasks.celery_app import get_worker_loop, get_worker_session_maker

logger = logging.getLogger(__name__)


def run_async_task(
    task_id: str,
    task_name: str,
    process_func: Callable[[AsyncSession], Awaitable[Any]],
) -> Any:
    loop = get_worker_loop()
    session_maker = get_worker_session_maker()

    async def execute():
        logger.info("%s_started task_id=%s", task_name, task_id)
        async with session_maker() as session:
            try:
                result = await process_func(session)
                await session.commit()
            except BaseException:
                await session.rollback()
                logger.exception("%s_failed task_id=%s", task_name, task_id)
                raise
        logger.info("%s_completed task_id=%s", task_name, task_id)
        return result

    return loop.run_until_complete(execute())
