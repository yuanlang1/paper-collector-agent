from __future__ import annotations

from sqlalchemy import Float, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.db.models.entity_base import EntityBase

PREFERENCE_SUBJECT = "__agent_preference__"


class MemoryFact(EntityBase, Base):
    __tablename__ = "memory_facts"
    __table_args__ = (
        UniqueConstraint("user_id", "dedupe_key", name="uq_memory_facts_user_dedupe"),
    )

    user_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    subject: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    dedupe_key: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    source_conversation_id: Mapped[str | None] = mapped_column(String(128))
    source_message_ids: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(16), default="active", nullable=False, index=True
    )
    confidence: Mapped[float | None] = mapped_column(Float)
