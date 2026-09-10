from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from functools import lru_cache

from app.config import settings
from app.models.memory import MemoryEpisode, MemoryFact, PREFERENCE_SUBJECT
from app.rag.index_construction.memory_episode_index_construction import (
    MemoryEpisodeIndexConstructionModule,
)
from app.rag.index_construction.memory_fact_index_construction import (
    MemoryFactIndexConstructionModule,
)

logger = logging.getLogger(__name__)


class MemoryIndexSynchronizer:
    def __init__(
        self,
        *,
        fact_index: MemoryFactIndexConstructionModule | None = None,
        episode_index: MemoryEpisodeIndexConstructionModule | None = None,
    ) -> None:
        self.fact_index = fact_index
        self.episode_index = episode_index

    async def upsert_facts(self, facts: Sequence[MemoryFact]) -> None:
        facts = [
            fact
            for fact in facts
            if fact.subject != PREFERENCE_SUBJECT
        ]
        if facts:
            await self._sync(
                "upsert facts", 
                lambda: self._fact_index().upsert_facts(facts)
            )

    async def delete_facts(self, fact_ids: Iterable[int]) -> None:
        fact_ids = list(fact_ids)
        if fact_ids:
            await self._sync(
                "delete facts",
                lambda: self._fact_index().delete_by_business_ids(fact_ids),
            )

    async def upsert_episodes(self, episodes: Sequence[MemoryEpisode]) -> None:
        if episodes:
            await self._sync(
                "upsert episodes",
                lambda: self._episode_index().upsert_episodes(episodes),
            )

    async def delete_episodes(self, episode_ids: Iterable[int]) -> None:
        episode_ids = list(episode_ids)
        if episode_ids:
            await self._sync(
                "delete episodes",
                lambda: self._episode_index().delete_by_business_ids(episode_ids),
            )

    @staticmethod
    async def _sync(
        action: str,
        operation: Callable[[], Awaitable[None]],
    ) -> None:
        if settings.MEMORY_RETRIEVAL_MODE != "hybrid":
            return
        try:
            await operation()
        except Exception:
            logger.warning("Unable to %s in Qdrant", action, exc_info=True)

    def _fact_index(self) -> MemoryFactIndexConstructionModule:
        if self.fact_index is None:
            self.fact_index = MemoryFactIndexConstructionModule()
        return self.fact_index

    def _episode_index(self) -> MemoryEpisodeIndexConstructionModule:
        if self.episode_index is None:
            self.episode_index = MemoryEpisodeIndexConstructionModule()
        return self.episode_index


@lru_cache(maxsize=1)
def get_memory_index_synchronizer() -> MemoryIndexSynchronizer:
    return MemoryIndexSynchronizer()
