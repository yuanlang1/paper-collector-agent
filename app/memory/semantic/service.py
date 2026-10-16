from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
import json

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MemoryFact
from app.db.repositories import memory_fact
from app.memory.errors import MemoryNotFoundError, MemoryValidationError
from app.memory.schemas import FactCandidate


class FactService:
    def __init__(
        self,
        db: AsyncSession,
        *,
        user_id: str,
    ) -> None:
        if not user_id.strip():
            raise MemoryValidationError("user_id is required")
        self.db = db
        self.user_id = user_id.strip()

    async def remember_explicit(
        self,
        *,
        subject: str,
        content: str,
        source_conversation_id: str | None,
    ) -> tuple[MemoryFact, bool]:
        subject, content = self._validate(subject, content)
        dedupe_key = self._dedupe_key(subject, content)

        async with self.db.begin():
            existing = await memory_fact.get_by_dedupe_key(
                self.db,
                user_id=self.user_id,
                dedupe_key=dedupe_key,
            )
            if existing is not None:
                if existing.status == "deleted":
                    await memory_fact.set_status(
                        self.db,
                        existing,
                        status="active",
                    )
                    await self.db.refresh(existing)
                    return existing, True
                return existing, False

            fact = await memory_fact.add(
                self.db,
                user_id=self.user_id,
                subject=subject,
                content=content,
                dedupe_key=dedupe_key,
                source="explicit",
                status="active",
                source_conversation_id=source_conversation_id,
                source_message_ids=None,
                confidence=1.0,
            )
            await self.db.refresh(fact)
            return fact, True

    async def stage_consolidated_candidate(
        self,
        *,
        candidate: FactCandidate,
        source_conversation_id: str,
        source_message_ids: list[int],
    ) -> MemoryFact | None:
        subject, content = self._validate(candidate.subject, candidate.content)
        dedupe_key = self._dedupe_key(subject, content)
        existing = await memory_fact.get_by_dedupe_key(
            self.db,
            user_id=self.user_id,
            dedupe_key=dedupe_key,
        )
        if existing is not None:
            return None
        return await memory_fact.add(
            self.db,
            user_id=self.user_id,
            subject=subject,
            content=content,
            dedupe_key=dedupe_key,
            source="consolidation",
            status="active",
            source_conversation_id=source_conversation_id,
            source_message_ids=json.dumps(source_message_ids),
            confidence=candidate.confidence,
        )

    async def list_active(self, *, limit: int = 50) -> Sequence[MemoryFact]:
        async with self.db.begin():
            return await memory_fact.list_facts(
                self.db,
                user_id=self.user_id,
                statuses=("active",),
                limit=self._limit(limit),
            )

    async def list_pending(self, *, limit: int = 50) -> Sequence[MemoryFact]:
        async with self.db.begin():
            return await memory_fact.list_facts(
                self.db,
                user_id=self.user_id,
                statuses=("pending",),
                limit=self._limit(limit),
            )

    async def list_active_by_subject(
        self,
        *,
        subject: str,
        limit: int = 50,
    ) -> Sequence[MemoryFact]:
        async with self.db.begin():
            return await memory_fact.list_by_subject(
                self.db,
                user_id=self.user_id,
                subject=subject.strip(),
                statuses=("active",),
                limit=self._limit(limit),
            )

    async def search_active(
        self,
        *,
        query: str,
        limit: int = 6,
    ) -> Sequence[MemoryFact]:
        if not query.strip():
            return []
        async with self.db.begin():
            return await memory_fact.search_active(
                self.db,
                user_id=self.user_id,
                query=query.strip(),
                limit=self._limit(limit),
            )

    async def get_active_by_ids(self, fact_ids: Sequence[int]) -> Sequence[MemoryFact]:
        async with self.db.begin():
            return await memory_fact.list_by_ids(
                self.db,
                user_id=self.user_id,
                fact_ids=fact_ids,
                statuses=("active",),
            )

    async def correct(
        self,
        *,
        fact_id: int,
        subject: str,
        content: str,
    ) -> MemoryFact:
        subject, content = self._validate(subject, content)
        dedupe_key = self._dedupe_key(subject, content)
        async with self.db.begin():
            fact = await self._get_editable(fact_id)
            duplicate = await memory_fact.get_by_dedupe_key(
                self.db,
                user_id=self.user_id,
                dedupe_key=dedupe_key,
            )
            if duplicate is not None and duplicate.id != fact.id:
                raise MemoryValidationError("An equivalent memory already exists")
            await memory_fact.update(
                self.db,
                fact,
                subject=subject,
                content=content,
                dedupe_key=dedupe_key,
            )
            await self.db.refresh(fact)
            return fact

    async def confirm(self, *, fact_id: int) -> MemoryFact:
        return await self._transition(
            fact_id=fact_id,
            from_status="pending",
            to_status="active",
        )

    async def reject(self, *, fact_id: int) -> MemoryFact:
        return await self._transition(
            fact_id=fact_id,
            from_status="pending",
            to_status="rejected",
        )

    async def forget(self, *, fact_id: int) -> MemoryFact:
        async with self.db.begin():
            fact = await self._get_editable(fact_id)
            await memory_fact.set_status(
                self.db,
                fact,
                status="deleted",
            )
            await self.db.refresh(fact)
            return fact

    async def _transition(
        self,
        *,
        fact_id: int,
        from_status: str,
        to_status: str,
    ) -> MemoryFact:
        async with self.db.begin():
            fact = await memory_fact.get(
                self.db,
                user_id=self.user_id,
                fact_id=fact_id,
                statuses=(from_status,),
            )
            if fact is None:
                raise MemoryNotFoundError("Memory fact not found")
            await memory_fact.set_status(
                self.db,
                fact,
                status=to_status,
            )
            await self.db.refresh(fact)
            return fact

    async def _get_editable(self, fact_id: int) -> MemoryFact:
        fact = await memory_fact.get(
            self.db,
            user_id=self.user_id,
            fact_id=fact_id,
            statuses=("active", "pending"),
        )
        if fact is None:
            raise MemoryNotFoundError("Memory fact not found")
        return fact

    @staticmethod
    def _validate(subject: str, content: str) -> tuple[str, str]:
        subject = subject.strip()
        content = content.strip()
        if not subject:
            raise MemoryValidationError("subject is required")
        if not content:
            raise MemoryValidationError("content is required")
        return subject, content

    @staticmethod
    def _dedupe_key(subject: str, content: str) -> str:
        value = f"{subject.casefold()}\n{content.casefold()}"
        return sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _limit(value: int) -> int:
        if not 1 <= value <= 50:
            raise MemoryValidationError("limit must be between 1 and 50")
        return value
