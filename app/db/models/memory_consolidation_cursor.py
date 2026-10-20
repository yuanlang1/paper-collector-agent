from __future__ import annotations

from sqlalchemy import BigInteger, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.db.models.entity_base import EntityBase


class MemoryConsolidationCursor(EntityBase, Base):
    __tablename__ = "memory_consolidation_cursors"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "conversation_id", name="uq_memory_cursors_user_conversation"
        ),
    )

    user_id: Mapped[str] = mapped_column(String(128))
    conversation_id: Mapped[str] = mapped_column(String(128))
    last_assistant_message_id: Mapped[int | None] = mapped_column(BigInteger)
