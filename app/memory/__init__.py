from __future__ import annotations

from sqlalchemy.orm import Session

from app.memory.episodic.service import EpisodeService
from app.memory.semantic.service import FactService


class Memory:
    """当前用户的长期记忆统一入口。"""

    def __init__(
        self,
        db: Session,
        *,
        user_id: str,
    ) -> None:
        self.db = db
        self.user_id = user_id
        self.facts = FactService(db, user_id=user_id)
        self.episodes = EpisodeService(db, user_id=user_id)