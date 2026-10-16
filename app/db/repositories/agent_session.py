from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentSession


class AgentSessionRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, *, user_id: int, thread_id: str, topic: str | None = None) -> AgentSession:
        session = AgentSession(
            user_id=user_id, thread_id=thread_id, topic=topic,
            creator_id=user_id, modifier_id=user_id,
        )
        self.db.add(session)
        await self.db.flush()
        return session

    async def get_by_thread_id(self, *, user_id: int, thread_id: str) -> AgentSession | None:
        return await self.db.scalar(select(AgentSession).where(
            AgentSession.user_id == user_id, AgentSession.thread_id == thread_id,
        ))

    async def require_session(self, *, user_id: int, thread_id: str) -> AgentSession:
        session = await self.get_by_thread_id(user_id=user_id, thread_id=thread_id)
        if session is None:
            raise ValueError("Session not found for user")
        return session

    async def list_by_user(self, *, user_id: int, limit: int = 50, offset: int = 0) -> list[AgentSession]:
        return list(await self.db.scalars(
            select(AgentSession).where(AgentSession.user_id == user_id)
            .order_by(AgentSession.modification_time.desc(), AgentSession.id.desc())
            .limit(limit).offset(offset)
        ))

    async def update_topic(self, *, user_id: int, thread_id: str, topic: str | None) -> AgentSession:
        session = await self.require_session(user_id=user_id, thread_id=thread_id)
        session.topic = topic
        session.modifier_id = user_id
        await self.db.flush()
        return session
