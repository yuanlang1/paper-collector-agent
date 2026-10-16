from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.memory.semantic.service import FactService
from app.db.models import MemoryFact, PREFERENCE_SUBJECT


class UserPreferenceService:
    def __init__(self, db: AsyncSession, *, user_id: str) -> None:
        self.facts = FactService(db, user_id=user_id)

    async def add(
        self,
        *,
        instruction: str,
        source_conversation_id: str | None,
    ) -> tuple[MemoryFact, bool]:
        return await self.facts.remember_explicit(
            subject=PREFERENCE_SUBJECT,
            content=instruction,
            source_conversation_id=source_conversation_id,
        )

    async def list_active(self, *, limit: int = 50) -> Sequence[MemoryFact]:
        return await self.facts.list_active_by_subject(
            subject=PREFERENCE_SUBJECT,
            limit=limit,
        )
