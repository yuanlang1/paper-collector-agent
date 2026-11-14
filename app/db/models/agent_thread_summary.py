from sqlalchemy import BigInteger, ForeignKey, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.db.models.entity_base import EntityBase


class AgentThreadSummary(EntityBase, Base):
    __tablename__ = "agent_thread_summary"
    __table_args__ = (
        UniqueConstraint("user_id", "thread_id", name="uq_agent_thread_summary_owner"),
    )

    user_id: Mapped[int] = mapped_column(BigInteger)
    thread_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("agent_thread.thread_id", ondelete="CASCADE")
    )
    content: Mapped[str] = mapped_column(Text)
    end_message_id: Mapped[str] = mapped_column(Text)
    version: Mapped[int] = mapped_column(BigInteger, server_default=text("1"))
