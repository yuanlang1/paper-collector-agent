from typing import Any
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.config import settings
from app.llm.graph.main.native_tools import get_tool_kind, requires_confirmation
from app.llm.graph.main.state import MainAgentState
from app.llm.streaming.tool_event import emit_custom_event
from app.llm.streaming.utils import content_to_text
from app.llm.tools.registry import ToolRegistry, build_tool_registry
from app.llm.subagents.registry import SubAgentRegistry


class SolveNode:
    def __init__(
        self,
        *,
        model: Any,
        tool_registry: ToolRegistry | None = None,
        subagent_registry: SubAgentRegistry,
    ) -> None:
        self.model = model
        self.tool_registry = tool_registry or build_tool_registry()
        self.subagent_registry = subagent_registry

    async def __call__(self, state: MainAgentState) -> dict:
        chunks = []
        reasoning_deltas: list[str] = []
        iteration = int(state.get("iteration_count") or 0) + 1
        reasoning_id = f"{state['run_id']}:solve:{uuid4().hex}"
        emit_custom_event(
            {
                "event": "iteration_started",
                "iteration": iteration,
            }
        )

        system = SystemMessage(content=str(state.get("system_context") or ""))

        messages = self._messages_from_window_start(
            state["messages"],
            conversation_window_start_id=state.get(
                "conversation_window_start_id"
            ),
        )
        messages = [
            *(
                [system]
                if system is not None
                else []
            ),
            *messages,
        ]

        async for chunk in self.model.astream(messages):
            chunks.append(chunk)

            reasoning_delta = chunk.additional_kwargs.get(
                "reasoning_content",
                "",
            )
            if reasoning_delta:
                reasoning_deltas.append(reasoning_delta)
                emit_custom_event(
                    {
                        "event": "reasoning_delta",
                        "delta": reasoning_delta,
                        "reasoning_id": reasoning_id,
                        "scope": "main",
                    }
                )

            content_delta = content_to_text(chunk.content)
            if content_delta:
                emit_custom_event(
                    {
                        "event": "content_delta",
                        "delta": content_delta,
                    }
                )

        assistant = chunks[0]
        for chunk in chunks[1:]:
            assistant += chunk
        reasoning_content = "".join(reasoning_deltas)

        if not assistant.tool_calls:
            return {
                "messages": [assistant],
                "reply": str(assistant.content or ""),
                "iteration_count": iteration,
                "reasoning_content": reasoning_content,
                "pending_tool_calls": [],
                "active_tool_call": None,
                "run_status": "completed",
                "error": None,
            }

        return {
            "messages": [assistant],
            "pending_tool_calls": [
                {
                    "id": call["id"],
                    "name": call["name"],
                    "args": call["args"],
                    "kind": get_tool_kind(
                        call["name"],
                        self.tool_registry,
                        self.subagent_registry,
                    ),
                    "requires_confirmation": requires_confirmation(
                        call["name"],
                        call["args"],
                        self.tool_registry,
                        self.subagent_registry,
                    ),
                }
                for call in assistant.tool_calls
            ],
            "active_tool_call": None,
            "iteration_count": iteration,
            "reasoning_content": reasoning_content,
            "run_status": "running",
        }

    @staticmethod
    def _messages_from_window_start(
        messages: list[BaseMessage],
        *,
        conversation_window_start_id: str | None,
    ) -> list[BaseMessage]:
        if conversation_window_start_id:
            for index, message in enumerate(messages):
                if (
                    message.id is not None
                    and str(message.id) == conversation_window_start_id
                ):
                    return messages[index:]

        return SolveNode._recent_conversation_messages(messages)

    @staticmethod
    def _recent_conversation_messages(
        messages: list[BaseMessage],
        *,
        history_turns: int | None = None,
    ) -> list[BaseMessage]:
        human_indexes = [
            index
            for index, message in enumerate(messages)
            if isinstance(message, HumanMessage)
        ]
        configured_turns = (
            settings.AGENT_HISTORY_TURNS
            if history_turns is None
            else history_turns
        )
        retained_turns = configured_turns + 1
        if len(human_indexes) <= retained_turns:
            return messages

        return messages[human_indexes[-retained_turns] :]
