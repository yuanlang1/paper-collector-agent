from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.db.models.entity_base import EntityBase


class SystemSetting(EntityBase, Base):
    __tablename__ = "config_setting"

    key: Mapped[str] = mapped_column(String(500), unique=True, index=True)
    value: Mapped[str | None] = mapped_column(Text)

    def __repr__(self):
        return f"<SystemSetting(key={self.key!r})>"
