from sqlalchemy import BigInteger, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.db.models.entity_base import EntityBase


class AgentThread(EntityBase, Base):
    __tablename__ = "agent_thread"
    __table_args__ = (
        Index("ix_agent_thread_user_modified", "user_id", "modification_time", "id"),
    )

    thread_id: Mapped[str] = mapped_column(String(128), unique=True)
    user_id: Mapped[int] = mapped_column(BigInteger)
    topic: Mapped[str | None] = mapped_column(String(500))
