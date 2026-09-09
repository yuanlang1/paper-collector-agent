from __future__ import annotations

from collections.abc import Callable
import logging

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from sqlalchemy.orm import Session

from app.config import settings
from app.llm.graph.main.state import MainAgentState
from app.llm.streaming.notify import langgraph_notifier
from app.llm.streaming.utils import content_to_text
from app.memory.schemas import MemoryUsage
from app.memory.soul import SoulLoadError
from app.runtime.system_context import SystemContextBuilder
from app.services.llm_profile_service import LlmRuntimeConfig


logger = logging.getLogger(__name__)


class MemoryNode:

    def __init__(
        self,
        *,
        db_factory: Callable[[], Session],
        system_context_builder: SystemContextBuilder,
        llm_config: LlmRuntimeConfig | None,
        memory_llm_config: LlmRuntimeConfig | None = None,
    ) -> None:
        self.db_factory = db_factory
        self.system_context_builder = system_context_builder
        self.llm_config = llm_config
        self.memory_llm_config = memory_llm_config

    async def __call__(
        self,
        state: MainAgentState,
        config: RunnableConfig | None = None,
    ) -> dict[str, object]:
        notify = langgraph_notifier(config).scoped(source="memory", node="memory")
        conversation_window_start_id = (
            self._conversation_window_start_id(state)
        )
        db: Session | None = None
        try:
            db = self.db_factory()
            result = await self.system_context_builder.build(
                user_message=self._latest_user_message(state),
                user_id=str(state.get("user_id") or "0"),
                conversation_id=str(state.get("conversation_id") or ""),
                db=db,
                llm_config=self.llm_config,
                memory_llm_config=self.memory_llm_config,
                on_memory_event=lambda event: notify(
                    str(event.get("event") or "memory_progress"),
                    {key: value for key, value in event.items() if key != "event"},
                ),
            )
            return {
                "conversation_window_start_id": conversation_window_start_id,
                "system_context": result.content,
                "memory_usage": result.memory_usage,
            }
        except SoulLoadError:
            raise
        except Exception:
            logger.exception("Memory node failed; continuing without retrieved memory")
            notify(
                "memory_retrieval_failed",
                {"facts_count": 0, "episodes_count": 0},
            )
            fallback_content = ""
            try:
                fallback = await self.system_context_builder.build(
                    user_message=self._latest_user_message(state),
                    user_id=str(state.get("user_id") or "0"),
                    conversation_id=str(state.get("conversation_id") or ""),
                    db=None,
                    llm_config=self.llm_config,
                    memory_llm_config=self.memory_llm_config,
                )
                fallback_content = fallback.content
            except Exception:
                logger.exception("Failed to build fallback system context")
            return {
                "conversation_window_start_id": conversation_window_start_id,
                "system_context": fallback_content,
                "memory_usage": self._failed_usage(),
            }
        finally:
            if db is not None:
                db.close()

    @staticmethod
    def _latest_user_message(state: MainAgentState) -> str:
        for message in reversed(state.get("messages") or []):
            if isinstance(message, HumanMessage):
                return content_to_text(message.content)
        return ""

    @staticmethod
    def _conversation_window_start_id(
        state: MainAgentState,
    ) -> str | None:
        messages = state.get("messages") or []
        if not messages:
            return None

        human_indexes = [
            index
            for index, message in enumerate(messages)
            if isinstance(message, HumanMessage)
        ]
        retained_human_messages = settings.AGENT_HISTORY_TURNS + 1
        start_index = (
            human_indexes[-retained_human_messages]
            if len(human_indexes) > retained_human_messages
            else 0
        )
        message_id = messages[start_index].id
        return str(message_id) if message_id else None

    @staticmethod
    def _failed_usage() -> MemoryUsage:
        return {
            "status": "failed",
            "facts_count": 0,
            "episodes_count": 0,
        }
