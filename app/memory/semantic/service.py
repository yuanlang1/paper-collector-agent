from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
import json

from sqlalchemy.orm import Session

from app.memory.errors import MemoryNotFoundError, MemoryValidationError
from app.memory.schemas import FactCandidate
from app.memory.semantic.repository import FactRepository
from app.models.memory import MemoryFact


class FactService:
    def __init__(
        self,
        db: Session,
        *,
        user_id: str,
        repository: FactRepository | None = None,
    ) -> None:
        if not user_id.strip():
            raise MemoryValidationError("user_id is required")

        self.db = db
        self.user_id = user_id.strip()
        self.repository = repository or FactRepository(db)

    def remember_explicit(
        self,
        *,
        subject: str,
        content: str,
        source_conversation_id: str | None,
    ) -> tuple[MemoryFact, bool]:
        subject, content = self._validate(subject, content)
        dedupe_key = self._dedupe_key(subject, content)

        try:
            existing = self.repository.get_by_dedupe_key(
                user_id=self.user_id,
                dedupe_key=dedupe_key,
            )
            if existing is not None:
                if existing.status == "deleted":
                    self.repository.set_status(existing, status="active")
                    self.db.commit()
                    self.db.refresh(existing)
                    return existing, True
                return existing, False

            fact = self.repository.add(
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
            self.db.commit()
            self.db.refresh(fact)
            return fact, True
        except Exception:
            self.db.rollback()
            raise

    def stage_consolidated_candidate(
        self,
        *,
        candidate: FactCandidate,
        source_conversation_id: str,
        source_message_ids: list[int],
    ) -> MemoryFact | None:
        """仅 flush；由 Consolidator 统一提交事务。"""
        subject, content = self._validate(candidate.subject, candidate.content)
        dedupe_key = self._dedupe_key(subject, content)

        existing = self.repository.get_by_dedupe_key(
            user_id=self.user_id,
            dedupe_key=dedupe_key,
        )
        if existing is not None:
            return None

        return self.repository.add(
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

    def list_active(self, *, limit: int = 50) -> Sequence[MemoryFact]:
        return self.repository.list(
            user_id=self.user_id,
            statuses=("active",),
            limit=self._limit(limit),
        )

    def list_pending(self, *, limit: int = 50) -> Sequence[MemoryFact]:
        return self.repository.list(
            user_id=self.user_id,
            statuses=("pending",),
            limit=self._limit(limit),
        )

    def search_active(self, *, query: str, limit: int = 6) -> Sequence[MemoryFact]:
        if not query.strip():
            return []
        return self.repository.search_active(
            user_id=self.user_id,
            query=query.strip(),
            limit=self._limit(limit),
        )

    def correct(
        self,
        *,
        fact_id: int,
        subject: str,
        content: str,
    ) -> MemoryFact:
        subject, content = self._validate(subject, content)
        fact = self._get_editable(fact_id)
        dedupe_key = self._dedupe_key(subject, content)

        duplicate = self.repository.get_by_dedupe_key(
            user_id=self.user_id,
            dedupe_key=dedupe_key,
        )
        if duplicate is not None and duplicate.id != fact.id:
            raise MemoryValidationError("An equivalent memory already exists")

        try:
            self.repository.update(
                fact,
                subject=subject,
                content=content,
                dedupe_key=dedupe_key,
            )
            self.db.commit()
            self.db.refresh(fact)
            return fact
        except Exception:
            self.db.rollback()
            raise

    def confirm(self, *, fact_id: int) -> MemoryFact:
        return self._transition(fact_id=fact_id, from_status="pending", to_status="active")

    def reject(self, *, fact_id: int) -> MemoryFact:
        return self._transition(fact_id=fact_id, from_status="pending", to_status="rejected")

    def forget(self, *, fact_id: int) -> MemoryFact:
        fact = self._get_editable(fact_id)
        try:
            self.repository.set_status(fact, status="deleted")
            self.db.commit()
            self.db.refresh(fact)
            return fact
        except Exception:
            self.db.rollback()
            raise

    def _transition(
        self,
        *,
        fact_id: int,
        from_status: str,
        to_status: str,
    ) -> MemoryFact:
        fact = self.repository.get(
            user_id=self.user_id,
            fact_id=fact_id,
            statuses=(from_status,),
        )
        if fact is None:
            raise MemoryNotFoundError("Memory fact not found")

        try:
            self.repository.set_status(fact, status=to_status)
            self.db.commit()
            self.db.refresh(fact)
            return fact
        except Exception:
            self.db.rollback()
            raise

    def _get_editable(self, fact_id: int) -> MemoryFact:
        fact = self.repository.get(
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
