from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging

from qdrant_client.http.exceptions import UnexpectedResponse

from app.rag.retrieval.memory_episode_retrieval import (
    MemoryEpisodeHybridRetrievalModule,
)
from app.rag.retrieval.memory_fact_retrieval import MemoryFactHybridRetrievalModule


logger = logging.getLogger(__name__)


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
            self._search_or_empty_when_collection_missing(
                collection_name=self.fact_retrieval.collection_name,
                search=lambda: self.fact_retrieval.search_for_user(
                    vectors,
                    user_id=user_id,
                    top_k=fact_candidate_limit,
                ),
            ),
            self._search_or_empty_when_collection_missing(
                collection_name=self.episode_retrieval.collection_name,
                search=lambda: self.episode_retrieval.search_for_conversation(
                    vectors,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    top_k=episode_candidate_limit,
                ),
            ),
        )
        return MemorySearchResult(
            fact_ids=[int(document.metadata["fact_id"]) for document in facts],
            episode_ids=[
                int(document.metadata["episode_id"])
                for document in episodes
            ],
        )

    @staticmethod
    async def _search_or_empty_when_collection_missing(
        *,
        collection_name: str,
        search,
    ):
        try:
            return await search()
        except UnexpectedResponse as exc:
            response_text = exc.content.decode("utf-8", errors="replace")
            if (
                exc.status_code == 404
                and f"Collection `{collection_name}`" in response_text
                and "doesn't exist" in response_text
            ):
                logger.info(
                    "Memory Qdrant collection is absent; treating it as empty: %s",
                    collection_name,
                )
                return []
            raise
