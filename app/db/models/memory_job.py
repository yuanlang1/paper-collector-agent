from datetime import datetime
from enum import StrEnum

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.db.models.entity_base import EntityBase


class MemoryJobKind(StrEnum):
    USERPROFILE = "userprofile"
    FACT = "fact"
    EPISODE = "episode"


class MemoryJobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class MemoryJob(EntityBase, Base):
    __tablename__ = "memory_job"
    __table_args__ = (
        CheckConstraint(
            "kind IN (" + ", ".join(f"'{kind.value}'" for kind in MemoryJobKind) + ")",
            name="ck_memory_job_kind",
        ),
        CheckConstraint(
            "status IN ("
            + ", ".join(f"'{status.value}'" for status in MemoryJobStatus)
            + ")",
            name="ck_memory_job_status",
        ),
        CheckConstraint("attempts >= 0", name="ck_memory_job_attempts"),
        CheckConstraint(
            "(kind = 'episode' AND source_thread_update_time IS NOT NULL) OR "
            "(kind <> 'episode' AND source_thread_update_time IS NULL)",
            name="ck_memory_job_episode_source",
        ),
        CheckConstraint(
            "(status = 'running' AND claim_token IS NOT NULL "
            "AND lease_expires_at IS NOT NULL) OR "
            "(status <> 'running' AND claim_token IS NULL "
            "AND lease_expires_at IS NULL)",
            name="ck_memory_job_lease",
        ),
        Index(
            "uq_memory_job_run_kind", "run_id", "kind", unique=True,
            postgresql_where=text("kind IN ('userprofile', 'fact')"),
        ),
        Index(
            "uq_memory_job_episode_source", "thread_id", "source_thread_update_time",
            unique=True, postgresql_where=text("kind = 'episode'"),
        ),
        Index(
            "uq_memory_job_active_episode", "thread_id", unique=True,
            postgresql_where=text("kind = 'episode' AND status IN ('pending', 'running')"),
        ),
        Index("ix_memory_job_dispatch", "status", "available_at", "id"),
        Index("ix_memory_job_thread_kind", "user_id", "thread_id", "kind", "id"),
    )

    job_id: Mapped[str] = mapped_column(String(128), unique=True)
    kind: Mapped[str] = mapped_column(String(32))
    user_id: Mapped[int] = mapped_column(BigInteger)
    thread_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("agent_thread.thread_id", ondelete="CASCADE")
    )
    run_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("agent_run.run_id", ondelete="CASCADE")
    )
    checkpoint_id: Mapped[str] = mapped_column(String(128))
    end_id: Mapped[str] = mapped_column(String(128))
    source_thread_update_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(
        String(32), default=MemoryJobStatus.PENDING, server_default=text("'pending'")
    )
    attempts: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    claim_token: Mapped[str | None] = mapped_column(String(128))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
