from sqlalchemy import BigInteger, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.models.entity_base import EntityBaseMixin


class AgentSession(EntityBaseMixin, Base):
    __tablename__ = "agent_session"
    __table_args__ = (
        Index("ix_agent_session_user_modified", "user_id", "modification_time", "id"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    thread_id: Mapped[str] = mapped_column(String(128, collation="utf8mb4_bin"), unique=True)
    user_id: Mapped[int] = mapped_column(BigInteger)
    topic: Mapped[str | None] = mapped_column(String(500))
