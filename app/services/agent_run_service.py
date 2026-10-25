import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any
from uuid import uuid4

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppException, ConflictException, NotFoundException
from app.db.models import MemoryJobKind, RunStatus
from app.db.repositories import agent_run, agent_thread, memory_job
from app.db.repositories.agent_run import AgentRunRow
from app.db.repositories.memory_job import MemoryJobRow
from app.llm import agent_build
from app.llm.agent_runner import AgentRunner
from app.llm.context import AgentRunContext
from app.tasks.modules.memory.extract_fact import extract_fact
from app.tasks.modules.memory.extract_userprofile import extract_userprofile

logger = logging.getLogger(__name__)


async def create_run(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str | None,
    metadata: dict[str, Any],
) -> AgentRunRow:
    try:
        async with db.begin():
            if thread_id:
                touched = await agent_thread.touch(
                    db,
                    user_id=user_id,
                    thread_id=thread_id,
                )
                if not touched:
                    raise NotFoundException("会话不存在")
            else:
                thread = await agent_thread.create(
                    db,
                    user_id=user_id,
                    thread_id=uuid4().hex,
                )
                thread_id = thread.thread_id
            run = await agent_run.create(
                db,
                user_id=user_id,
                thread_id=thread_id,
                run_id=f"run_{uuid4().hex}",
                metadata=metadata,
            )
            return run
    except IntegrityError as exc:
        if exc.orig.diag.constraint_name == "uq_agent_run_active_thread":
            raise ConflictException("该会话已有待执行或执行中的请求") from exc
        raise


async def _update_status(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    run_id: str,
    status: RunStatus,
    error_message: str | None = None,
) -> AgentRunRow | None:
    if status == RunStatus.PENDING:
        raise ValueError("Cannot transition a run back to pending")

    expected: tuple[RunStatus, ...] = (
        (RunStatus.PENDING,) if status == RunStatus.RUNNING else (RunStatus.RUNNING,)
    )

    if status == RunStatus.CANCELLED:
        expected = (RunStatus.PENDING, RunStatus.RUNNING)
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


async def update_status(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    run_id: str,
    status: RunStatus,
    error_message: str | None = None,
) -> AgentRunRow | None:
    async with db.begin():
        return await _update_status(
            db,
            user_id=user_id,
            thread_id=thread_id,
            run_id=run_id,
            status=status,
            error_message=error_message,
        )


async def complete_run(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    run_id: str,
    checkpoint_id: str,
    end_id: str,
) -> tuple[AgentRunRow | None, list[MemoryJobRow]]:
    if not checkpoint_id or not end_id:
        raise ValueError("Completed run requires a checkpoint and end message")
    async with db.begin():
        run = await _update_status(
            db,
            user_id=user_id,
            thread_id=thread_id,
            run_id=run_id,
            status=RunStatus.COMPLETED,
        )
        if run is None:
            return None, []
        jobs = await memory_job.create_run_jobs(
            db,
            user_id=user_id,
            thread_id=thread_id,
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            end_id=end_id,
        )
        return run, jobs


async def get_run(
    db: AsyncSession,
    *,
    user_id: int,
    thread_id: str,
    run_id: str,
) -> AgentRunRow:
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
) -> list[AgentRunRow]:
    async with db.begin():
        runs = await agent_run.list_by_owned_thread(
            db,
            user_id=user_id,
            thread_id=thread_id,
            limit=limit,
            offset=offset,
        )
        if runs is None:
            raise NotFoundException("会话不存在")
        return runs


async def run(
    *,
    db: AsyncSession,
    run: AgentRunRow,
    input: dict[str, Any],
    checkpointer,
    store,
) -> AsyncIterator[dict[str, Any]]:
    identity = {
        "thread_id": run.thread_id,
        "run_id": run.run_id,
        "user_id": run.user_id,
    }
    started = await update_status(db, **identity, status=RunStatus.RUNNING)

    if started is None:
        raise ConflictException("该请求已结束或已开始执行")

    try:
        if "resume" not in input:
            yield {"type": "status", "status": RunStatus.RUNNING}

        metadata = run.run_metadata
        model = await agent_build.get_model(
            db,
            model_id=metadata["model_id"],
            user_id=run.user_id,
            think_effort=metadata.get("think_effort"),
        )
        tool_registry = agent_build.build_tools(db, user_id=run.user_id, model=model)

        runner = AgentRunner(
            context=AgentRunContext(
                thread_id=run.thread_id,
                run_id=run.run_id,
                user_id=str(run.user_id),
            ),
            model=model,
            tool_registry=tool_registry,
            checkpointer=checkpointer,
            store=store,
        )

        async with aclosing(runner.run(input)) as stream:
            async for event in stream:
                yield event
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
            "type": "error",
            "status": RunStatus.FAILED,
            "message": exc.message
            if isinstance(exc, AppException)
            else "大模型调用失败，请稍后重试",
        }
        return

    if status == RunStatus.COMPLETED:
        finished, jobs = await complete_run(
            db,
            **identity,
            checkpoint_id=snapshot.config["configurable"]["checkpoint_id"],
            end_id=snapshot.values["messages"][-1].id,
        )
        for job in jobs:
            try:
                if job.kind == MemoryJobKind.USERPROFILE:
                    await asyncio.to_thread(extract_userprofile.delay, job.job_id)
                elif job.kind == MemoryJobKind.FACT:
                    await asyncio.to_thread(extract_fact.delay, job.job_id)
            except Exception:
                logger.warning(
                    "Memory dispatch failed; pending job will be retried: job_id=%s",
                    job.job_id,
                )
    else:
        finished = await update_status(db, **identity, status=status)
    if finished is None:
        raise ConflictException("该请求已结束")
    interrupts = [{"id": item.id, "value": item.value} for item in snapshot.interrupts]
    if status == RunStatus.INTERRUPTED:
        yield {
            "type": "interrupt",
            "status": status,
            "interrupts": interrupts,
        }
    yield {
        "type": "done",
        "status": status,
        "interrupts": interrupts,
    }
