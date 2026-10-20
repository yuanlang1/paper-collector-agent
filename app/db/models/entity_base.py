from datetime import datetime

from sqlalchemy import BigInteger, DateTime, func, text
from sqlalchemy.orm import Mapped, mapped_column


class EntityBase:
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    creator_id: Mapped[int] = mapped_column(
        BigInteger, server_default=text("0"), nullable=False
    )
    creation_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    modifier_id: Mapped[int] = mapped_column(
        BigInteger, server_default=text("0"), nullable=False
    )
    modification_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
