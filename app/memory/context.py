from __future__ import annotations

from sqlalchemy.orm import Session

from app.memory.episodic.service import EpisodeService
from app.memory.retrieval_gate import MemoryRetrievalGate
from app.memory.semantic.service import FactService
from app.services.llm_profile_service import LlmRuntimeConfig


class MemoryContextService:
    def __init__(
        self,
        db: Session,
        *,
        user_id: str,
        retrieval_gate: MemoryRetrievalGate | None = None,
        max_characters: int = 1_500,
    ) -> None:
        self.user_id = user_id
        self.max_characters = max_characters
        self.fact_service = FactService(db, user_id=user_id)
        self.episode_service = EpisodeService(db, user_id=user_id)
        self.retrieval_gate = retrieval_gate or MemoryRetrievalGate()

    async def build_context(
        self,
        *,
        user_message: str,
        conversation_id: str,
        llm_config: LlmRuntimeConfig | None,
    ) -> str:
        decision = await self.retrieval_gate.decide(
            user_message=user_message,
            llm_config=llm_config,
        )
        if not decision.retrieve:
            return ""

        query = decision.query or user_message
        facts = self.fact_service.search_active(query=query, limit=6)

        episodes = self.episode_service.list_active_for_conversation(
            conversation_id=conversation_id,
            limit=2,
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

        return "\n\n".join(sections)[: self.max_characters]