from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class MemoryFact(Base):
    __tablename__ = "memory_facts"
    __table_args__ = (
        UniqueConstraint("user_id", "dedupe_key", name="uq_memory_facts_user_dedupe"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class MemoryEpisode(Base):
    __tablename__ = "memory_episodes"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "source_assistant_message_id",
            name="uq_memory_episodes_user_source_message",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    happened_at: Mapped[date] = mapped_column(Date, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    source_conversation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_assistant_message_id: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default="pending", nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class MemoryConsolidationCursor(Base):
    __tablename__ = "memory_consolidation_cursors"

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    last_assistant_message_id: Mapped[int | None] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )