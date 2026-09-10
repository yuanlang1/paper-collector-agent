from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.memory.episodic.service import EpisodeService
from app.memory.retrieval_gate import MemoryRetrievalGate
from app.memory.schemas import (
    MemoryContextResult,
    MemoryRetrievalStatus,
    MemoryUsage,
)
from app.memory.semantic.service import FactService
from app.models.memory import MemoryEpisode, MemoryFact
from app.rag.retrieval.memory_retrieval import MemoryHybridRetrieval
from app.services.llm_profile_service import LlmRuntimeConfig


MemoryEventCallback = Callable[[dict[str, Any]], None]
logger = logging.getLogger(__name__)


class MemoryContextService:
    def __init__(
        self,
        db: Session,
        *,
        user_id: str,
        retrieval_gate: MemoryRetrievalGate | None = None,
        memory_retrieval: MemoryHybridRetrieval | None = None,
        max_characters: int = 1_500,
    ) -> None:
        self.user_id = user_id
        self.max_characters = max_characters
        self.fact_service = FactService(db, user_id=user_id)
        self.episode_service = EpisodeService(db, user_id=user_id)
        self.retrieval_gate = retrieval_gate or MemoryRetrievalGate()
        self.memory_retrieval = memory_retrieval

    async def build_context(
        self,
        *,
        user_message: str,
        conversation_id: str,
        llm_config: LlmRuntimeConfig | None,
        on_event: MemoryEventCallback | None = None,
    ) -> MemoryContextResult:
        decision = await self.retrieval_gate.decide(
            user_message=user_message,
            llm_config=llm_config,
        )
        if not decision.retrieve:
            return MemoryContextResult(
                content="",
                usage=self._usage("skipped"),
            )

        self._emit(
            on_event,
            {
                "event": "memory_retrieval_started",
                "facts_count": 0,
                "episodes_count": 0,
            },
        )

        query = decision.query or user_message
        facts, episodes = await self._retrieve_records(
            query=query,
            conversation_id=conversation_id,
        )

        sections: list[str] = []

        if facts:
            sections.append(
                "[已确认的长期偏好与研究背景]\n"
                + "\n".join(
                    f"- {fact.subject}：{fact.content}"
                    for fact in facts
                )
            )

        if episodes:
            sections.append(
                "[当前会话已确认摘要]\n"
                + "\n".join(
                    f"- {episode.happened_at.isoformat()}：{episode.summary}"
                    for episode in episodes
                )
            )

        content = "\n\n".join(sections)[: self.max_characters]
        usage = self._usage(
            "completed" if content else "empty",
            facts_count=len(facts),
            episodes_count=len(episodes),
        )
        self._emit(
            on_event,
            {
                "event": f"memory_retrieval_{usage['status']}",
                "facts_count": usage["facts_count"],
                "episodes_count": usage["episodes_count"],
            },
        )
        return MemoryContextResult(content=content, usage=usage)

    async def _retrieve_records(
        self,
        *,
        query: str,
        conversation_id: str,
    ) -> tuple[list[MemoryFact], list[MemoryEpisode]]:
        if settings.MEMORY_RETRIEVAL_MODE == "hybrid":
            try:
                retrieval = self.memory_retrieval or MemoryHybridRetrieval()
                self.memory_retrieval = retrieval
                result = await retrieval.retrieve(
                    query=query,
                    user_id=self.user_id,
                    conversation_id=conversation_id,
                )
                return (
                    self._order_by_ids(
                        self.fact_service.get_active_by_ids(result.fact_ids),
                        result.fact_ids,
                        limit=6,
                    ),
                    self._order_by_ids(
                        self.episode_service.get_active_by_ids_for_conversation(
                            episode_ids=result.episode_ids,
                            conversation_id=conversation_id,
                        ),
                        result.episode_ids,
                        limit=2,
                    ),
                )
            except Exception:
                logger.warning(
                    "Hybrid memory retrieval failed; falling back to SQL",
                    exc_info=True,
                )

        return (
            list(self.fact_service.search_active(query=query, limit=6)),
            list(
                self.episode_service.list_active_for_conversation(
                    conversation_id=conversation_id,
                    limit=2,
                )
            ),
        )

    @staticmethod
    def _order_by_ids(
        records: Any,
        ids: list[int],
        *,
        limit: int,
    ) -> list[Any]:
        records_by_id = {record.id: record for record in records}
        return [
            records_by_id[record_id]
            for record_id in ids
            if record_id in records_by_id
        ][:limit]

    @staticmethod
    def _usage(
        status: MemoryRetrievalStatus,
        *,
        facts_count: int = 0,
        episodes_count: int = 0,
    ) -> MemoryUsage:
        return {
            "status": status,
            "facts_count": facts_count,
            "episodes_count": episodes_count,
        }

    @staticmethod
    def _emit(
        callback: MemoryEventCallback | None,
        payload: dict[str, Any],
    ) -> None:
        if callback is not None:
            callback(payload)
