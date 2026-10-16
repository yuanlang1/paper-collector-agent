from datetime import datetime

from sqlalchemy import BigInteger, DateTime, FetchedValue, text
from sqlalchemy.orm import Mapped, mapped_column


class EntityBaseMixin:
    # Fetch database-generated timestamps during flush, avoiding async lazy loads.
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    creator_id: Mapped[int] = mapped_column(BigInteger, server_default=text("0"), nullable=False)
    creation_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False,
    )
    modifier_id: Mapped[int] = mapped_column(BigInteger, server_default=text("0"), nullable=False)
    modification_time: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"),
        server_onupdate=FetchedValue(),
        nullable=False,
    )
