from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from app.common.exceptions import ConflictException, NotFoundException
from app.db.repositories import agent_run, agent_thread


@dataclass
class ThreadInfo:
    thread_id: str
    topic: str | None
    create_time: datetime
    update_time: datetime


async def get_threads(
    db, user_id: int, limit: int = 20, offset: int = 0
) -> list[ThreadInfo]:
    async with db.begin():
        threads = await agent_thread.list_by_user(
            db,
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

        return [
            ThreadInfo(
                thread_id=thread.thread_id,
                topic=thread.topic,
                create_time=thread.creation_time,
                update_time=thread.modification_time,
            )
            for thread in threads
        ]


async def create_thread(db, user_id: int, message: str | None = None) -> ThreadInfo:
    thread_id = uuid4().hex
    topic = message[:40] if message is not None else None

    async with db.begin():
        thread = await agent_thread.create(
            db, user_id=user_id, thread_id=thread_id, topic=topic
        )

        return ThreadInfo(
            thread_id=thread_id,
            topic=topic,
            create_time=thread.creation_time,
            update_time=thread.modification_time,
        )


async def get_history(db, *, user_id: int, thread_id: str, checkpointer):
    async with db.begin():
        thread = await agent_thread.get_by_thread_id(
            db,
            user_id=user_id,
            thread_id=thread_id,
        )
        if thread is None:
            raise NotFoundException("会话不存在")

    checkpoint = await checkpointer.aget_tuple(
        {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""},
        }
    )

    return checkpoint.checkpoint["channel_values"] if checkpoint is not None else []


async def delete_thread(db, user_id: int, thread_id: str) -> None:
    async with db.begin():
        thread = await agent_thread.get_by_thread_id(
            db,
            user_id=user_id,
            thread_id=thread_id,
            for_update=True,
        )

        if thread is None:
            raise NotFoundException("会话不存在")

        if await agent_run.has_active(db, user_id=user_id, thread_id=thread_id):
            raise ConflictException("该会话有待执行或执行中的请求，暂时不能删除")

        await agent_thread.delete(
            db,
            user_id=user_id,
            thread_id=thread_id,
        )
