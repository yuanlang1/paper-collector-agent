import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any
from uuid import uuid4

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppException, ConflictException, NotFoundException
from app.db.models import AgentRun, RunStatus
from app.db.repositories import agent_run, agent_thread
from app.llm import agent_build
from app.llm.agent_runner import AgentRunner

logger = logging.getLogger(__name__)


async def create_run(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str | None,
    metadata: dict[str, Any],
) -> AgentRun:
    try:
        async with db.begin():
            if thread_id:
                thread = await agent_thread.get_by_thread_id(
                    db,
                    user_id=user_id,
                    thread_id=thread_id,
                    for_update=True,
                )
                if thread is None:
                    raise NotFoundException("会话不存在")
                await agent_thread.touch(db, user_id=user_id, thread_id=thread_id)
            else:
                thread = await agent_thread.create(
                    db, user_id=user_id, thread_id=uuid4().hex,
                )
            run = await agent_run.create(
                db,
                user_id=user_id,
                thread_id=thread.thread_id,
                run_id=f"run_{uuid4().hex}",
                metadata=metadata,
            )
            return run
    except IntegrityError as exc:
        if exc.orig.diag.constraint_name == "uq_agent_run_active_thread":
            raise ConflictException("该会话已有待执行或执行中的请求") from exc
        raise


async def update_status(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    run_id: str,
    status: RunStatus,
    error_message: str | None = None,
) -> AgentRun | None:
    if status == RunStatus.PENDING:
        raise ValueError("Cannot transition a run back to pending")
    expected = (
        (RunStatus.PENDING,) if status == RunStatus.RUNNING else (RunStatus.RUNNING,)
    )
    if status == RunStatus.CANCELLED:
        expected = (RunStatus.PENDING, RunStatus.RUNNING)
    async with db.begin():
        thread = await agent_thread.get_by_thread_id(
            db,
            user_id=user_id,
            thread_id=thread_id,
            for_update=True,
        )
        if thread is None:
            return None
        run = await agent_run.update_status(
            db,
            user_id=user_id,
            thread_id=thread_id,
            run_id=run_id,
            status=status,
            expected_status=expected,
            error_message=error_message,
        )
        if run is not None:
            await agent_thread.touch(db, user_id=user_id, thread_id=thread_id)
        return run


async def get_run(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    run_id: str,
) -> AgentRun:
    async with db.begin():
        run = await agent_run.get_by_run_id(
            db,
            user_id=user_id,
            thread_id=thread_id,
            run_id=run_id,
        )
        if run is None:
            raise NotFoundException("执行记录不存在")
        return run


async def list_runs(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    limit: int = 20,
    offset: int = 0,
) -> list[AgentRun]:
    async with db.begin():
        thread = await agent_thread.get_by_thread_id(
            db, user_id=user_id, thread_id=thread_id
        )

        if thread is None:
            raise NotFoundException("会话不存在")
        return await agent_run.list_by_thread(
            db,
            user_id=user_id,
            thread_id=thread_id,
            limit=limit,
            offset=offset,
        )


async def run(
    *,
    db: AsyncSession,
    run: AgentRun,
    input: dict[str, Any],
    checkpointer,
    store,
) -> AsyncIterator[dict[str, Any]]:
    fields = {"thread_id": run.thread_id, "run_id": run.run_id}
    identity = {**fields, "user_id": run.user_id}
    started = await update_status(db, **identity, status=RunStatus.RUNNING)

    if started is None:
        raise ConflictException("该请求已结束或已开始执行")

    try:
        yield {**fields, "type": "status", "status": RunStatus.RUNNING}

        metadata = run.run_metadata
        model = await agent_build.get_model(
            db,
            model_id=metadata["model_id"],
            user_id=run.user_id,
            think_effort=metadata.get("think_effort"),
        )
        tool_registry = agent_build.build_tools(db, user_id=run.user_id)

        runner = AgentRunner(
            thread_id=run.thread_id,
            run_id=run.run_id,
            user_id=str(run.user_id),
            model=model,
            tool_registry=tool_registry,
            checkpointer=checkpointer,
            store=store,
        )

        async with aclosing(runner.run(input)) as stream:
            async for event in stream:
                yield {**event, **fields}
        snapshot = await runner.get_state()
        errors = [task.error for task in snapshot.tasks if task.error]
        if errors:
            raise RuntimeError(str(errors[0]))
        if snapshot.interrupts:
            status = RunStatus.INTERRUPTED
        elif not snapshot.next:
            status = RunStatus.COMPLETED
        else:
            raise RuntimeError("Graph stopped with unfinished tasks and no interrupt")
    except Exception as exc:
        logger.exception("Agent run failed: run_id=%s", run.run_id)
        await update_status(
            db,
            **identity,
            status=RunStatus.FAILED,
            error_message=str(exc) or type(exc).__name__,
        )
        yield {
            **fields,
            "type": "error",
            "status": RunStatus.FAILED,
            "message": exc.message
            if isinstance(exc, AppException)
            else "大模型调用失败，请稍后重试",
        }
        return

    finished = await update_status(db, **identity, status=status)
    if finished is None:
        raise ConflictException("该请求已结束")
    interrupts = [
        {"id": item.id, "value": item.value} for item in snapshot.interrupts
    ]
    if status == RunStatus.INTERRUPTED:
        yield {
            **fields,
            "type": "interrupt",
            "status": status,
            "interrupts": interrupts,
        }
    yield {
        **fields,
        "type": "done",
        "status": status,
        "interrupts": interrupts,
    }
