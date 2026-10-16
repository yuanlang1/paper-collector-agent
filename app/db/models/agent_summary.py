from sqlalchemy import BigInteger, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.models.entity_base import EntityBaseMixin


class AgentSummary(EntityBaseMixin, Base):
    __tablename__ = "agent_summary"
    __table_args__ = (
        UniqueConstraint("thread_id", "start_id", "end_id", name="uq_agent_summary_range"),
        Index("ix_agent_summary_thread_end", "thread_id", "end_id"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    thread_id: Mapped[str] = mapped_column(
        String(128, collation="utf8mb4_bin"), ForeignKey("agent_session.thread_id"),
    )
    start_id: Mapped[int] = mapped_column(BigInteger)
    end_id: Mapped[int] = mapped_column(BigInteger)
    summary: Mapped[str] = mapped_column(Text)
