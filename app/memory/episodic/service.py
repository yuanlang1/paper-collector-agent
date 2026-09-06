from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from sqlalchemy.orm import Session

from app.memory.errors import MemoryNotFoundError, MemoryValidationError
from app.memory.episodic.repository import EpisodeRepository
from app.memory.schemas import EpisodeCandidate
from app.models.memory import MemoryEpisode


class EpisodeService:
    def __init__(
        self,
        db: Session,
        *,
        user_id: str,
        repository: EpisodeRepository | None = None,
    ) -> None:
        if not user_id.strip():
            raise MemoryValidationError("user_id is required")

        self.db = db
        self.user_id = user_id.strip()
        self.repository = repository or EpisodeRepository(db)

    def stage_consolidated_candidate(
        self,
        *,
        candidate: EpisodeCandidate,
        source_conversation_id: str,
    ) -> MemoryEpisode | None:
        summary = candidate.summary.strip()
        if not summary:
            raise MemoryValidationError("episode summary is required")

        existing = self.repository.get_by_source_message(
            user_id=self.user_id,
            assistant_message_id=candidate.source_assistant_message_id,
        )
        if existing is not None:
            return None

        return self.repository.add(
            user_id=self.user_id,
            happened_at=candidate.happened_at,
            summary=summary,
            source_conversation_id=source_conversation_id,
            source_assistant_message_id=candidate.source_assistant_message_id,
            status="active",
        )

    def list_active(self, *, limit: int = 50) -> Sequence[MemoryEpisode]:
        return self.repository.list(
            user_id=self.user_id,
            statuses=("active",),
            limit=self._limit(limit),
        )

    def list_pending(self, *, limit: int = 50) -> Sequence[MemoryEpisode]:
        return self.repository.list(
            user_id=self.user_id,
            statuses=("pending",),
            limit=self._limit(limit),
        )

    def confirm(self, *, episode_id: int) -> MemoryEpisode:
        return self._transition(episode_id, "pending", "active")

    def reject(self, *, episode_id: int) -> MemoryEpisode:
        return self._transition(episode_id, "pending", "rejected")

    def forget(self, *, episode_id: int) -> MemoryEpisode:
        episode = self.repository.get(
            user_id=self.user_id,
            episode_id=episode_id,
            statuses=("active", "pending"),
        )
        if episode is None:
            raise MemoryNotFoundError("Memory episode not found")

        try:
            self.repository.set_status(episode, status="deleted")
            self.db.commit()
            self.db.refresh(episode)
            return episode
        except Exception:
            self.db.rollback()
            raise

    def _transition(
        self,
        episode_id: int,
        from_status: str,
        to_status: str,
    ) -> MemoryEpisode:
        episode = self.repository.get(
            user_id=self.user_id,
            episode_id=episode_id,
            statuses=(from_status,),
        )
        if episode is None:
            raise MemoryNotFoundError("Memory episode not found")

        try:
            self.repository.set_status(episode, status=to_status)
            self.db.commit()
            self.db.refresh(episode)
            return episode
        except Exception:
            self.db.rollback()
            raise

    def list_active_for_conversation(
        self,
        *,
        conversation_id: str,
        limit: int = 2,
    ) -> Sequence[MemoryEpisode]:
        return self.repository.list_by_conversation(
            user_id=self.user_id,
            conversation_id=conversation_id,
            statuses=("active",),
            limit=self._limit(limit),
        )

    @staticmethod
    def _limit(value: int) -> int:
        if not 1 <= value <= 50:
            raise MemoryValidationError("limit must be between 1 and 50")
        return value
