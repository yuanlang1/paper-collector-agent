from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models.memory import MemoryFact


class FactRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get(
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
        return self.db.scalar(statement)

    def get_by_dedupe_key(
        self,
        *,
        user_id: str,
        dedupe_key: str,
    ) -> MemoryFact | None:
        return self.db.scalar(
            select(MemoryFact).where(
                MemoryFact.user_id == user_id,
                MemoryFact.dedupe_key == dedupe_key,
            )
        )

    def add(
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
        self.db.flush()
        return fact

    def list(
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
        return list(self.db.scalars(statement))

    def search_active(
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
        return list(self.db.scalars(statement))

    def update(
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
        self.db.flush()
        return fact

    def set_status(
        self,
        fact: MemoryFact,
        *,
        status: str,
    ) -> MemoryFact:
        fact.status = status
        self.db.flush()
        return fact