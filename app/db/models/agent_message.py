from typing import Any

from sqlalchemy import ForeignKey, Index, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.models.entity_base import EntityBaseMixin


class AgentMessage(EntityBaseMixin, Base):
    __tablename__ = "agent_message"
    __table_args__ = (
        Index("ix_agent_message_thread_id", "thread_id", "id"),
        Index("ix_agent_message_run_id", "run_id"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    run_id: Mapped[str] = mapped_column(String(128, collation="utf8mb4_bin"))
    thread_id: Mapped[str] = mapped_column(
        String(128, collation="utf8mb4_bin"), ForeignKey("agent_session.thread_id"),
    )
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str | list[Any]] = mapped_column(JSON)
    message_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
