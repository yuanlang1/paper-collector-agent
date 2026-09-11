from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.rag.retrieval.memory_episode_retrieval import (
    MemoryEpisodeHybridRetrievalModule,
)
from app.rag.retrieval.memory_fact_retrieval import MemoryFactHybridRetrievalModule


@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    fact_ids: list[int]
    episode_ids: list[int]


class MemoryHybridRetrieval:
    def __init__(
        self,
        *,
        fact_retrieval: MemoryFactHybridRetrievalModule | None = None,
        episode_retrieval: MemoryEpisodeHybridRetrievalModule | None = None,
    ) -> None:
        self.fact_retrieval = fact_retrieval or MemoryFactHybridRetrievalModule()
        self.episode_retrieval = episode_retrieval or MemoryEpisodeHybridRetrievalModule()

    async def retrieve(
        self,
        *,
        query: str,
        user_id: str,
        conversation_id: str,
        fact_candidate_limit: int = 12,
        episode_candidate_limit: int = 6,
    ) -> MemorySearchResult:
        vectors = await self.fact_retrieval.encode_query(query)
        facts, episodes = await asyncio.gather(
            self.fact_retrieval.search_for_user(
                vectors,
                user_id=user_id,
                top_k=fact_candidate_limit,
            ),
            self.episode_retrieval.search_for_conversation(
                vectors,
                user_id=user_id,
                conversation_id=conversation_id,
                top_k=episode_candidate_limit,
            ),
        )
        return MemorySearchResult(
            fact_ids=[int(document.metadata["fact_id"]) for document in facts],
            episode_ids=[
                int(document.metadata["episode_id"])
                for document in episodes
            ],
        )
