from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MemoryEpisode


async def get(
    db: AsyncSession,
    *,
    user_id: str,
    episode_id: int,
    statuses: tuple[str, ...] | None = None,
) -> MemoryEpisode | None:
    statement = select(MemoryEpisode).where(
        MemoryEpisode.id == episode_id,
        MemoryEpisode.user_id == user_id,
    )
    if statuses is not None:
        statement = statement.where(MemoryEpisode.status.in_(statuses))
    return await db.scalar(statement)


async def get_by_source_message(
    db: AsyncSession,
    *,
    user_id: str,
    assistant_message_id: int,
) -> MemoryEpisode | None:
    return await db.scalar(
        select(MemoryEpisode).where(
            MemoryEpisode.user_id == user_id,
            MemoryEpisode.source_assistant_message_id == assistant_message_id,
        )
    )


async def add(
    db: AsyncSession,
    *,
    user_id: str,
    happened_at: date,
    summary: str,
    source_conversation_id: str,
    source_assistant_message_id: int,
    status: str,
) -> MemoryEpisode:
    episode = MemoryEpisode(
        user_id=user_id,
        happened_at=happened_at,
        summary=summary,
        source_conversation_id=source_conversation_id,
        source_assistant_message_id=source_assistant_message_id,
        status=status,
    )
    db.add(episode)
    await db.flush()
    return episode


async def list_episodes(
    db: AsyncSession,
    *,
    user_id: str,
    statuses: tuple[str, ...],
    limit: int,
) -> Sequence[MemoryEpisode]:
    statement = (
        select(MemoryEpisode)
        .where(
            MemoryEpisode.user_id == user_id,
            MemoryEpisode.status.in_(statuses),
        )
        .order_by(MemoryEpisode.happened_at.desc(), MemoryEpisode.id.desc())
        .limit(limit)
    )
    return list(await db.scalars(statement))


async def set_status(
    db: AsyncSession,
    episode: MemoryEpisode,
    *,
    status: str,
) -> MemoryEpisode:
    episode.status = status
    await db.flush()
    return episode


async def list_by_conversation(
    db: AsyncSession,
    *,
    user_id: str,
    conversation_id: str,
    statuses: tuple[str, ...],
    limit: int,
) -> Sequence[MemoryEpisode]:
    statement = (
        select(MemoryEpisode)
        .where(
            MemoryEpisode.user_id == user_id,
            MemoryEpisode.source_conversation_id == conversation_id,
            MemoryEpisode.status.in_(statuses),
        )
        .order_by(
            MemoryEpisode.happened_at.desc(),
            MemoryEpisode.id.desc(),
        )
        .limit(limit)
    )
    return list(await db.scalars(statement))


async def list_by_ids_for_conversation(
    db: AsyncSession,
    *,
    user_id: str,
    episode_ids: Sequence[int],
    conversation_id: str,
    statuses: tuple[str, ...],
) -> Sequence[MemoryEpisode]:
    if not episode_ids:
        return []
    return list(
        await db.scalars(
            select(MemoryEpisode).where(
                MemoryEpisode.id.in_(episode_ids),
                MemoryEpisode.user_id == user_id,
                MemoryEpisode.source_conversation_id == conversation_id,
                MemoryEpisode.status.in_(statuses),
            )
        )
    )
