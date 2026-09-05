from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from sqlalchemy.orm import Session as DbSession

from app.llm.streaming.utils import encode_sse


@dataclass
class Session:
    conversation_id: str
    db: DbSession
    user_id: str = "0"
    run_id: str | None = None
    assistant_message_id: int | None = None
    message: str | None = None
    resume_payload: dict[str, Any] | None = None
    llm_profile: dict[str, Any] | None = None
    memory_llm_profile: dict[str, Any] | None = None
    paper_search_source_limits: dict[str, int] | None = None
    event_sequence: int = 0

    @classmethod
    def create(
        cls,
        *,
        message: str,
        conversation_id: str,
        db: DbSession,
        user_id: str = "0",
        run_id: str | None = None,
        assistant_message_id: int | None = None,
        llm_profile: dict[str, Any] | None = None,
        memory_llm_profile: dict[str, Any] | None = None,
        paper_search_source_limits: dict[str, int] | None = None,
    ) -> "Session":
        return cls(
            conversation_id=conversation_id,
            run_id=run_id or f"run_{uuid4().hex}",
            assistant_message_id=assistant_message_id,
            message=message,
            db=db,
            user_id=user_id,
            llm_profile=llm_profile,
            memory_llm_profile=memory_llm_profile,
            paper_search_source_limits=paper_search_source_limits,
        )

    @classmethod
    def for_resume_lookup(
        cls,
        *,
        conversation_id: str,
        db: DbSession,
        user_id: str = "0",
    ) -> "Session":
        return cls(
            conversation_id=conversation_id,
            db=db,
            user_id=user_id,
        )

    @classmethod
    def resume(
        cls,
        *,
        conversation_id: str,
        run_id: str,
        resume_payload: dict[str, Any],
        db: DbSession,
        user_id: str = "0",
        assistant_message_id: int | None = None,
        llm_profile: dict[str, Any] | None = None,
        memory_llm_profile: dict[str, Any] | None = None,
    ) -> "Session":
        return cls(
            conversation_id=conversation_id,
            run_id=run_id,
            resume_payload=resume_payload,
            assistant_message_id=assistant_message_id,
            db=db,
            user_id=user_id,
            llm_profile=llm_profile,
            memory_llm_profile=memory_llm_profile,
        )

    @property
    def graph_config(self) -> dict[str, Any]:
        configurable: dict[str, Any] = {
            "thread_id": self.conversation_id,
        }
        metadata: dict[str, Any] = {
            "conversation_id": self.conversation_id,
            "graph": "solve_tool_v1",
        }

        if self.run_id:
            configurable["run_id"] = self.run_id
            metadata["run_id"] = self.run_id

        return {
            "configurable": configurable,
            "metadata": metadata,
            "recursion_limit": 50,
        }

    @property
    def graph_input(self) -> dict[str, Any] | Command:
        if self.resume_payload is not None:
            return Command(resume=self.resume_payload)

        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "run_id": self.run_id,
            "llm_profile": self.llm_profile,
            "memory_llm_profile": self.memory_llm_profile,
            "paper_search_request": None,
            "paper_search_handoff": None,
            "paper_search_tool_call_id": None,
            "paper_search_source_limits": self.paper_search_source_limits,
            "task_review_request": None,
            "task_review_handoff": None,
            "task_review_tool_call_id": None,
            "messages": [HumanMessage(content=self.message)],
            "conversation_window_start_id": None,
            "system_context": "",
            "memory_usage": None,
            "pending_tool_calls": [],
            "active_tool_call": None,
            "iteration_count": 0,
            "reasoning_content": "",
            "last_action_result": None,
            "artifact_refs": [],
            "reply": "",
            "run_status": "running",
            "error": None,
        }

    def build_sse_envelope(
        self,
        event: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        self.event_sequence += 1

        return {
            "event_id": f"{self.run_id}:{self.event_sequence}",
            "sequence": self.event_sequence,
            "event": event,
            "conversation_id": self.conversation_id,
            "run_id": self.run_id,
            "llm_profile": self.llm_profile,
            "assistant_message_id": self.assistant_message_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }

    @staticmethod
    def encode_sse_envelope(
        envelope: dict[str, Any],
    ) -> str:
        return encode_sse(
            str(envelope["event"]),
            envelope,
        )

    def encode_sse(
        self,
        event: str,
        data: dict[str, Any],
    ) -> str:
        return self.encode_sse_envelope(
            self.build_sse_envelope(event, data),
        )
