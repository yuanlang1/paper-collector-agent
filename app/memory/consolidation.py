from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.memory.episodic.service import EpisodeService
from app.memory.schemas import (
    ConsolidationOutput,
    ConsolidationResult,
    HistoryTurn,
)
from app.memory.semantic.service import FactService
from app.models.memory import MemoryConsolidationCursor


class HistoryReader(Protocol):
    async def list_completed_turns_after(
        self,
        *,
        user_id: str,
        conversation_id: str,
        after_assistant_message_id: int | None,
        limit: int,
    ) -> list[HistoryTurn]:
        ...


class MemoryExtractionModel(Protocol):
    async def extract_memory(
        self,
        *,
        turns: Sequence[HistoryTurn],
    ) -> ConsolidationOutput:
        ...


class Consolidator:
    def __init__(
        self,
        db: Session,
        *,
        user_id: str,
        history_reader: HistoryReader,
        extraction_model: MemoryExtractionModel,
        threshold_turns: int = 6,
    ) -> None:
        self.db = db
        self.user_id = user_id
        self.history_reader = history_reader
        self.extraction_model = extraction_model
        self.threshold_turns = threshold_turns
        self._locks: dict[str, asyncio.Lock] = {}

    async def consolidate_if_due(
        self,
        *,
        conversation_id: str,
    ) -> ConsolidationResult:
        lock = self._locks.setdefault(conversation_id, asyncio.Lock())

        async with lock:
            cursor = self._get_cursor(conversation_id)
            turns = await self.history_reader.list_completed_turns_after(
                user_id=self.user_id,
                conversation_id=conversation_id,
                after_assistant_message_id=(
                    cursor.last_assistant_message_id if cursor else None
                ),
                limit=self.threshold_turns,
            )

            if len(turns) < self.threshold_turns:
                return ConsolidationResult(
                    due=False,
                    facts_created=0,
                    episode_created=False,
                    last_assistant_message_id=(
                        cursor.last_assistant_message_id if cursor else None
                    ),
                )

            extracted = await self.extraction_model.extract_memory(turns=turns)
            fact_service = FactService(self.db, user_id=self.user_id)
            episode_service = EpisodeService(self.db, user_id=self.user_id)

            try:
                created_facts = 0
                source_message_ids = [
                    turn.assistant_message_id
                    for turn in turns
                ]

                for candidate in extracted.facts:
                    created = fact_service.stage_consolidated_candidate(
                        candidate=candidate,
                        source_conversation_id=conversation_id,
                        source_message_ids=source_message_ids,
                    )
                    if created is not None:
                        created_facts += 1

                created_episode = False
                if extracted.episode is not None:
                    created_episode = (
                        episode_service.stage_consolidated_candidate(
                            candidate=extracted.episode,
                            source_conversation_id=conversation_id,
                        )
                        is not None
                    )

                last_message_id = turns[-1].assistant_message_id
                self._advance_cursor(
                    conversation_id=conversation_id,
                    last_assistant_message_id=last_message_id,
                )
                self.db.commit()

                return ConsolidationResult(
                    due=True,
                    facts_created=created_facts,
                    episode_created=created_episode,
                    last_assistant_message_id=last_message_id,
                )
            except Exception:
                self.db.rollback()
                raise

    def _get_cursor(
        self,
        conversation_id: str,
    ) -> MemoryConsolidationCursor | None:
        return self.db.scalar(
            select(MemoryConsolidationCursor).where(
                MemoryConsolidationCursor.user_id == self.user_id,
                MemoryConsolidationCursor.conversation_id == conversation_id,
            )
        )

    def _advance_cursor(
        self,
        *,
        conversation_id: str,
        last_assistant_message_id: int,
    ) -> None:
        cursor = self._get_cursor(conversation_id)
        if cursor is None:
            cursor = MemoryConsolidationCursor(
                user_id=self.user_id,
                conversation_id=conversation_id,
                last_assistant_message_id=last_assistant_message_id,
            )
            self.db.add(cursor)
        else:
            cursor.last_assistant_message_id = last_assistant_message_id
        self.db.flush()