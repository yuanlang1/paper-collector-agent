from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MemoryConsolidationCursor


class ConsolidationCursorRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get(self, *, user_id: str, conversation_id: str) -> MemoryConsolidationCursor | None:
        return await self.db.scalar(
            select(MemoryConsolidationCursor).where(
                MemoryConsolidationCursor.user_id == user_id,
                MemoryConsolidationCursor.conversation_id == conversation_id,
            )
        )

    async def advance(
        self, *, user_id: str, conversation_id: str, last_assistant_message_id: int,
    ) -> None:
        cursor = await self.get(user_id=user_id, conversation_id=conversation_id)
        if cursor is None:
            self.db.add(MemoryConsolidationCursor(
                user_id=user_id,
                conversation_id=conversation_id,
                last_assistant_message_id=last_assistant_message_id,
            ))
        else:
            cursor.last_assistant_message_id = last_assistant_message_id
        await self.db.flush()
