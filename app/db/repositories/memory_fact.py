from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MemoryFact


class FactRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get(
        self,
        *,
        user_id: str,
        fact_id: int,
        statuses: tuple[str, ...] | None = None,
    ) -> MemoryFact | None:
        statement = select(MemoryFact).where(
            MemoryFact.id == fact_id,
            MemoryFact.user_id == user_id,
        )
        if statuses is not None:
            statement = statement.where(MemoryFact.status.in_(statuses))
        return await self.db.scalar(statement)

    async def get_by_dedupe_key(
        self,
        *,
        user_id: str,
        dedupe_key: str,
    ) -> MemoryFact | None:
        return await self.db.scalar(
            select(MemoryFact).where(
                MemoryFact.user_id == user_id,
                MemoryFact.dedupe_key == dedupe_key,
            )
        )

    async def add(
        self,
        *,
        user_id: str,
        subject: str,
        content: str,
        dedupe_key: str,
        source: str,
        status: str,
        source_conversation_id: str | None,
        source_message_ids: str | None,
        confidence: float | None,
    ) -> MemoryFact:
        fact = MemoryFact(
            user_id=user_id,
            subject=subject,
            content=content,
            dedupe_key=dedupe_key,
            source=source,
            status=status,
            source_conversation_id=source_conversation_id,
            source_message_ids=source_message_ids,
            confidence=confidence,
        )
        self.db.add(fact)
        await self.db.flush()
        return fact

    async def list(
        self,
        *,
        user_id: str,
        statuses: tuple[str, ...],
        limit: int,
    ) -> Sequence[MemoryFact]:
        statement = (
            select(MemoryFact)
            .where(
                MemoryFact.user_id == user_id,
                MemoryFact.status.in_(statuses),
            )
            .order_by(MemoryFact.updated_at.desc(), MemoryFact.id.desc())
            .limit(limit)
        )
        return list(await self.db.scalars(statement))

    async def list_by_subject(
        self,
        *,
        user_id: str,
        subject: str,
        statuses: tuple[str, ...],
        limit: int,
    ) -> Sequence[MemoryFact]:
        statement = (
            select(MemoryFact)
            .where(
                MemoryFact.user_id == user_id,
                MemoryFact.subject == subject,
                MemoryFact.status.in_(statuses),
            )
            .order_by(MemoryFact.updated_at.desc(), MemoryFact.id.desc())
            .limit(limit)
        )
        return list(await self.db.scalars(statement))

    async def list_by_ids(
        self,
        *,
        user_id: str,
        fact_ids: Sequence[int],
        statuses: tuple[str, ...],
    ) -> Sequence[MemoryFact]:
        if not fact_ids:
            return []
        return list(
            await self.db.scalars(
                select(MemoryFact).where(
                    MemoryFact.id.in_(fact_ids),
                    MemoryFact.user_id == user_id,
                    MemoryFact.status.in_(statuses),
                )
            )
        )

    async def search_active(
        self,
        *,
        user_id: str,
        query: str,
        limit: int,
    ) -> Sequence[MemoryFact]:
        escaped = (
            query.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        pattern = f"%{escaped}%"

        statement = (
            select(MemoryFact)
            .where(
                MemoryFact.user_id == user_id,
                MemoryFact.status == "active",
                or_(
                    MemoryFact.subject.ilike(pattern, escape="\\"),
                    MemoryFact.content.ilike(pattern, escape="\\"),
                ),
            )
            .order_by(MemoryFact.updated_at.desc(), MemoryFact.id.desc())
            .limit(limit)
        )
        return list(await self.db.scalars(statement))

    async def update(
        self,
        fact: MemoryFact,
        *,
        subject: str,
        content: str,
        dedupe_key: str,
    ) -> MemoryFact:
        fact.subject = subject
        fact.content = content
        fact.dedupe_key = dedupe_key
        await self.db.flush()
        return fact

    async def set_status(
        self,
        fact: MemoryFact,
        *,
        status: str,
    ) -> MemoryFact:
        fact.status = status
        await self.db.flush()
        return fact
