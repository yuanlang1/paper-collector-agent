import unittest
from datetime import date
from unittest.mock import Mock

from app.memory.episodic.service import EpisodeService
from app.memory.schemas import EpisodeCandidate, FactCandidate
from app.memory.semantic.service import FactService


class MemoryAutoActivationTests(unittest.TestCase):
    def test_consolidated_fact_is_active_immediately(self):
        repository = Mock()
        repository.get_by_dedupe_key.return_value = None
        repository.add.return_value = object()
        service = FactService(
            Mock(),
            user_id="0",
            repository=repository,
        )

        service.stage_consolidated_candidate(
            candidate=FactCandidate(
                subject="检索偏好",
                content="优先开放获取论文",
            ),
            source_conversation_id="conv-1",
            source_message_ids=[10, 11],
        )

        self.assertEqual(repository.add.call_args.kwargs["status"], "active")

    def test_consolidated_episode_is_active_immediately(self):
        repository = Mock()
        repository.get_by_source_message.return_value = None
        repository.add.return_value = object()
        service = EpisodeService(
            Mock(),
            user_id="0",
            repository=repository,
        )

        service.stage_consolidated_candidate(
            candidate=EpisodeCandidate(
                happened_at=date(2026, 9, 2),
                summary="确认了开放获取优先的检索策略。",
                source_assistant_message_id=11,
            ),
            source_conversation_id="conv-1",
        )

        self.assertEqual(repository.add.call_args.kwargs["status"], "active")


if __name__ == "__main__":
    unittest.main()
