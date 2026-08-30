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
    run_id: str | None = None
    message: str | None = None
    resume_payload: dict[str, Any] | None = None
    event_sequence: int = 0

    @classmethod
    def create(
        cls,
        *,
        message: str,
        conversation_id: str,
        db: DbSession,
    ) -> "Session":
        return cls(
            conversation_id=conversation_id,
            run_id=f"run_{uuid4().hex}",
            message=message,
            db=db,
        )

    @classmethod
    def for_resume_lookup(
        cls,
        *,
        conversation_id: str,
        db: DbSession,
    ) -> "Session":
        return cls(
            conversation_id=conversation_id,
            db=db,
        )

    @classmethod
    def resume(
        cls,
        *,
        conversation_id: str,
        run_id: str,
        resume_payload: dict[str, Any],
        db: DbSession,
    ) -> "Session":
        return cls(
            conversation_id=conversation_id,
            run_id=run_id,
            resume_payload=resume_payload,
            db=db,
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
            "run_id": self.run_id,
            "paper_search_request": None,
            "paper_search_handoff": None,
            "paper_search_tool_call_id": None,
            "task_review_request": None,
            "task_review_handoff": None,
            "task_review_tool_call_id": None,
            "messages": [HumanMessage(content=self.message)],
            "pending_tool_calls": [],
            "active_tool_call": None,
            "reasoning_content": "",
            "last_action_result": None,
            "artifact_refs": [],
            "reply": "",
            "run_status": "running",
            "error": None,
        }

    def encode_sse(
        self,
        event: str,
        data: dict[str, Any],
    ) -> str:
        self.event_sequence += 1

        return encode_sse(
            event,
            {
                "event_id": f"{self.run_id}:{self.event_sequence}",
                "sequence": self.event_sequence,
                "event": event,
                "conversation_id": self.conversation_id,
                "run_id": self.run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": data,
            },
        )
