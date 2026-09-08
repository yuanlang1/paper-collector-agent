from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.orm import Session

from app.memory.semantic.service import FactService
from app.models.memory import MemoryFact


PREFERENCE_SUBJECT = "__agent_preference__"


class UserPreferenceService:
    """Stores durable per-user agent preferences in the existing fact store."""

    def __init__(self, db: Session, *, user_id: str) -> None:
        self.facts = FactService(db, user_id=user_id)

    def add(
        self,
        *,
        instruction: str,
        source_conversation_id: str | None,
    ) -> tuple[MemoryFact, bool]:
        return self.facts.remember_explicit(
            subject=PREFERENCE_SUBJECT,
            content=instruction,
            source_conversation_id=source_conversation_id,
        )

    def list_active(self, *, limit: int = 50) -> Sequence[MemoryFact]:
        return self.facts.list_active_by_subject(
            subject=PREFERENCE_SUBJECT,
            limit=limit,
        )
