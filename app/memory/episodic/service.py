from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MemoryEpisode
from app.db.repositories import memory_episode
from app.memory.errors import MemoryNotFoundError, MemoryValidationError
from app.memory.schemas import EpisodeCandidate


class EpisodeService:
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

    async def stage_consolidated_candidate(
        self,
        *,
        candidate: EpisodeCandidate,
        source_conversation_id: str,
    ) -> MemoryEpisode | None:
        summary = candidate.summary.strip()
        if not summary:
            raise MemoryValidationError("episode summary is required")
        existing = await memory_episode.get_by_source_message(
            self.db,
            user_id=self.user_id,
            assistant_message_id=candidate.source_assistant_message_id,
        )
        if existing is not None:
            return None
        return await memory_episode.add(
            self.db,
            user_id=self.user_id,
            happened_at=candidate.happened_at,
            summary=summary,
            source_conversation_id=source_conversation_id,
            source_assistant_message_id=candidate.source_assistant_message_id,
            status="active",
        )

    async def list_active(self, *, limit: int = 50) -> Sequence[MemoryEpisode]:
        async with self.db.begin():
            return await memory_episode.list_episodes(
                self.db,
                user_id=self.user_id,
                statuses=("active",),
                limit=self._limit(limit),
            )

    async def list_pending(self, *, limit: int = 50) -> Sequence[MemoryEpisode]:
        async with self.db.begin():
            return await memory_episode.list_episodes(
                self.db,
                user_id=self.user_id,
                statuses=("pending",),
                limit=self._limit(limit),
            )

    async def confirm(self, *, episode_id: int) -> MemoryEpisode:
        return await self._transition(
            episode_id,
            "pending",
            "active",
        )

    async def reject(self, *, episode_id: int) -> MemoryEpisode:
        return await self._transition(
            episode_id,
            "pending",
            "rejected",
        )

    async def forget(self, *, episode_id: int) -> MemoryEpisode:
        async with self.db.begin():
            episode = await memory_episode.get(
                self.db,
                user_id=self.user_id,
                episode_id=episode_id,
                statuses=("active", "pending"),
            )
            if episode is None:
                raise MemoryNotFoundError("Memory episode not found")
            await memory_episode.set_status(
                self.db,
                episode,
                status="deleted",
            )
            await self.db.refresh(episode)
            return episode

    async def _transition(
        self,
        episode_id: int,
        from_status: str,
        to_status: str,
    ) -> MemoryEpisode:
        async with self.db.begin():
            episode = await memory_episode.get(
                self.db,
                user_id=self.user_id,
                episode_id=episode_id,
                statuses=(from_status,),
            )
            if episode is None:
                raise MemoryNotFoundError("Memory episode not found")
            await memory_episode.set_status(
                self.db,
                episode,
                status=to_status,
            )
            await self.db.refresh(episode)
            return episode

    async def list_active_for_conversation(
        self,
        *,
        conversation_id: str,
        limit: int = 2,
    ) -> Sequence[MemoryEpisode]:
        async with self.db.begin():
            return await memory_episode.list_by_conversation(
                self.db,
                user_id=self.user_id,
                conversation_id=conversation_id,
                statuses=("active",),
                limit=self._limit(limit),
            )

    async def get_active_by_ids_for_conversation(
        self,
        *,
        episode_ids: Sequence[int],
        conversation_id: str,
    ) -> Sequence[MemoryEpisode]:
        async with self.db.begin():
            return await memory_episode.list_by_ids_for_conversation(
                self.db,
                user_id=self.user_id,
                episode_ids=episode_ids,
                conversation_id=conversation_id,
                statuses=("active",),
            )

    @staticmethod
    def _limit(value: int) -> int:
        if not 1 <= value <= 50:
            raise MemoryValidationError("limit must be between 1 and 50")
        return value
