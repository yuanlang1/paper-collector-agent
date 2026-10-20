from sqlalchemy import BigInteger, Boolean, String, Text, false
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.db.models.entity_base import EntityBase


class LlmSetting(EntityBase, Base):
    __tablename__ = "llm_setting"

    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    provider: Mapped[str] = mapped_column(String(64))
    base_url: Mapped[str] = mapped_column(String(2048))
    model: Mapped[str] = mapped_column(String(255))
    api_key: Mapped[str] = mapped_column(Text)
    is_small_model: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=false(), nullable=False
    )
