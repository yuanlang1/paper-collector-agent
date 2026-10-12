from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class _AgentEntityBase:
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
    )
    creator_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    creation_time: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )
    modifier_id: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    modification_time: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class AgentConversation(_AgentEntityBase, Base):
    __tablename__ = "agent_conversation"

    conversation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    thread_id: Mapped[str | None] = mapped_column(String(128))
    title: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), nullable=False)


class AgentRun(_AgentEntityBase, Base):
    __tablename__ = "agent_run"

    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    input_message_id: Mapped[int | None] = mapped_column(BigInteger().with_variant(Integer, "sqlite"))
    output_message_id: Mapped[int | None] = mapped_column(BigInteger().with_variant(Integer, "sqlite"))
    pending_interrupt_id: Mapped[str | None] = mapped_column(String(128))
    pending_interrupt_payload: Mapped[str | None] = mapped_column(Text)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[str | None] = mapped_column("metadata", Text)


class AgentMessage(_AgentEntityBase, Base):
    __tablename__ = "agent_message"

    conversation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
