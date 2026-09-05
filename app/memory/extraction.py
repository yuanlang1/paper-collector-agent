from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.model_factory import create_validated_structured_chat_model
from app.llm.structured_output import ValidatedJsonInvoker
from app.memory.schemas import (
    ConsolidationOutput,
    EpisodeCandidate,
    FactCandidate,
    HistoryTurn,
    MemoryExtractionPayload,
)


SYSTEM_PROMPT = """
    You distill completed conversations from a literature research assistant.

    Extract only durable user preferences, research scope, inclusion/exclusion
    criteria, or decisions that remain useful in a later conversation. Also write
    at most one concise, factual episode describing what happened in this batch.

    Do not retain small talk, temporary requests, unverified claims, credentials,
    personal secrets, paper full text, or raw tool output. Return no more than
    three facts. If nothing is worth retaining, return an empty facts list and a
    null episode_summary.
"""


class LangChainMemoryExtractor:
    def __init__(
        self,
        invoker: ValidatedJsonInvoker[MemoryExtractionPayload] | None = None,
        *,
        max_characters: int = 12_000,
    ) -> None:
        self.invoker = invoker or create_validated_structured_chat_model(
            MemoryExtractionPayload,
            temperature=0,
            max_attempts=2,
            purpose="memory",
        )
        self.max_characters = max_characters

    async def extract_memory(
        self,
        *,
        turns: Sequence[HistoryTurn],
    ) -> ConsolidationOutput:
        if not turns:
            return ConsolidationOutput(facts=[], episode=None)

        payload = await self.invoker.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=self._format_turns(turns)),
            ]
        )
        facts = [
            FactCandidate(
                subject=item.subject.strip(),
                content=item.content.strip(),
                confidence=item.confidence,
            )
            for item in payload.facts
            if item.subject.strip() and item.content.strip()
        ]

        summary = (payload.episode_summary or "").strip()
        last_turn = turns[-1]
        episode = (
            EpisodeCandidate(
                happened_at=last_turn.completed_at.date(),
                summary=summary,
                source_assistant_message_id=last_turn.assistant_message_id,
            )
            if summary
            else None
        )
        return ConsolidationOutput(facts=facts, episode=episode)

    def _format_turns(self, turns: Sequence[HistoryTurn]) -> str:
        chunks: list[str] = []
        used = 0

        for turn in turns:
            block = (
                f"User: {turn.user_content.strip()}\n"
                f"Assistant: {turn.assistant_content.strip()}"
            )
            remaining = self.max_characters - used
            if remaining <= 0:
                break
            if len(block) > remaining:
                block = block[:remaining]
            chunks.append(block)
            used += len(block)
            if used >= self.max_characters:
                break

        return "\n\n".join(chunks)
