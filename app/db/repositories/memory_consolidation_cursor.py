from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MemoryConsolidationCursor


async def get(
    db: AsyncSession,
    *,
    user_id: str,
    conversation_id: str,
) -> MemoryConsolidationCursor | None:
    return await db.scalar(
        select(MemoryConsolidationCursor).where(
            MemoryConsolidationCursor.user_id == user_id,
            MemoryConsolidationCursor.conversation_id == conversation_id,
        )
    )


async def advance(
    db: AsyncSession,
    *,
    user_id: str,
    conversation_id: str,
    last_assistant_message_id: int,
) -> None:
    cursor = await get(
        db,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    if cursor is None:
        db.add(
            MemoryConsolidationCursor(
                user_id=user_id,
                conversation_id=conversation_id,
                last_assistant_message_id=last_assistant_message_id,
            )
        )
    else:
        cursor.last_assistant_message_id = last_assistant_message_id
    await db.flush()
