from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentMessage, AgentSummary
from app.db.repositories.agent_session import AgentSessionRepository


class AgentSummaryRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def save(
        self, *, user_id: int, thread_id: str, start_id: int, end_id: int, summary: str,
    ) -> AgentSummary:
        if start_id > end_id:
            raise ValueError("start_id must not exceed end_id")
        await AgentSessionRepository(self.db).require_session(user_id=user_id, thread_id=thread_id)
        endpoints = set(await self.db.scalars(select(AgentMessage.id).where(
            AgentMessage.thread_id == thread_id, AgentMessage.id.in_({start_id, end_id}),
        )))
        if endpoints != {start_id, end_id}:
            raise ValueError("Summary endpoints must belong to the session")
        statement = insert(AgentSummary).values(
            thread_id=thread_id, start_id=start_id, end_id=end_id, summary=summary,
            creator_id=user_id, modifier_id=user_id,
        )
        await self.db.execute(statement.on_duplicate_key_update(
            summary=summary, modifier_id=user_id, modification_time=func.now(),
        ))
        await self.db.flush()
        return await self.db.scalar(select(AgentSummary).where(
            AgentSummary.thread_id == thread_id,
            AgentSummary.start_id == start_id, AgentSummary.end_id == end_id,
        ).execution_options(populate_existing=True))

    async def get_by_range(
        self, *, user_id: int, thread_id: str, start_id: int, end_id: int,
    ) -> AgentSummary | None:
        await AgentSessionRepository(self.db).require_session(user_id=user_id, thread_id=thread_id)
        return await self.db.scalar(select(AgentSummary).where(
            AgentSummary.thread_id == thread_id,
            AgentSummary.start_id == start_id, AgentSummary.end_id == end_id,
        ))

    async def get_latest(self, *, user_id: int, thread_id: str) -> AgentSummary | None:
        rows = await self.list_by_thread(user_id=user_id, thread_id=thread_id, limit=1)
        return rows[0] if rows else None

    async def list_by_thread(
        self, *, user_id: int, thread_id: str, limit: int = 50, offset: int = 0,
    ) -> list[AgentSummary]:
        await AgentSessionRepository(self.db).require_session(user_id=user_id, thread_id=thread_id)
        return list(await self.db.scalars(select(AgentSummary).where(
            AgentSummary.thread_id == thread_id,
        ).order_by(AgentSummary.end_id.desc(), AgentSummary.id.desc()).limit(limit).offset(offset)))
