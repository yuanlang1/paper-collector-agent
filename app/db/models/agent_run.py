from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.db.models.entity_base import EntityBase


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


class AgentRun(EntityBase, Base):
    __tablename__ = "agent_run"
    __table_args__ = (
        CheckConstraint(
            "status IN ("
            + ", ".join(f"'{status.value}'" for status in RunStatus)
            + ")",
            name="ck_agent_run_status",
        ),
        Index("ix_agent_run_user_thread_id", "user_id", "thread_id", "id"),
        Index(
            "uq_agent_run_active_thread",
            "thread_id",
            unique=True,
            postgresql_where=text("status IN ('pending', 'running')"),
        ),
    )

    run_id: Mapped[str] = mapped_column(String(128), unique=True)
    thread_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("agent_thread.thread_id")
    )
    user_id: Mapped[int] = mapped_column(BigInteger)
    status: Mapped[str] = mapped_column(
        String(32),
        default=RunStatus.PENDING,
        server_default=text("'pending'"),
    )
    run_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
