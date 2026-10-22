from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import UTC, datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel

from app.common.exceptions import BadRequestException
from app.llm.context import BaseAgentContext
from app.llm.loop.agent_loop import AgentLoop
from app.llm.tools.registry import ToolRegistry

_EVENT_TRACE_FIELDS = {
    "thread_id", "run_id", "session_id", "user_id", "app_run_id",
    "parent_thread_id", "parent_run_id", "parent_session_id",
    "tool_use_id", "attempt", "attempts", "namespace", "agent_scope", "ts",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


class AgentRunner:
    def __init__(
        self,
        *,
        context: BaseAgentContext,
        model: BaseChatModel,
        tool_registry: ToolRegistry,
        checkpointer,
        store,
        system_prompt: str | None = "你是一个助手",
    ) -> None:
        self.context = context
        self.checkpointer = checkpointer
        self.store = store
        self.model = model
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.config = {
            "configurable": {"thread_id": context.thread_id},
            "metadata": {"user_id": context.user_id, "app_run_id": context.run_id},
        }

    def _chunk_text(self, chunk: Any) -> str:
        if chunk is None:
            return ""

        content = getattr(chunk, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part.get("text") or "")
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts)
        return str(content) if content is not None else ""

    async def get_state(self):
        return await self.agent.graph.aget_state(self.config)

    async def run(self, input: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        self.agent = AgentLoop(
            context=self.context,
            model=self.model,
            tools=self.tool_registry.get_tools(),
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
            store=self.store,
            name="agent",
        )

        snapshot = await self.get_state()
        resuming = "resume" in input
        if resuming and not snapshot.interrupts:
            raise BadRequestException("该会话没有待恢复的中断")
        if snapshot.interrupts and not resuming:
            raise BadRequestException("该会话正在等待恢复，请传递 resume")

        messages = (
            Command(resume=input["resume"])
            if resuming
            else [HumanMessage(content=input["message"])]
        )

        if not resuming:
            yield {"type": "status", "message": "正在思考…"}

        async with aclosing(
            self.agent.stream(
                messages=messages,
                config=self.config,
            )
        ) as stream:
            async for part in stream:
                if part["type"] == "custom":
                    payload = part["data"]
                    if isinstance(payload, BaseModel):
                        payload = payload.model_dump(mode="json")
                    payload = {
                        key: value for key, value in payload.items()
                        if key not in _EVENT_TRACE_FIELDS
                    }
                    if payload.get("type") in ("tool_start", "tool_end", "tool_failed"):
                        payload["label"] = self.tool_registry.get_preview_name(payload["name"])
                    yield payload

                elif part["type"] == "messages":
                    chunk, meta = part["data"]
                    if not isinstance(chunk, AIMessage):
                        continue
                    if meta.get("langgraph_node") not in (None, "solve"):
                        continue
                    text = self._chunk_text(chunk)
                    if text:
                        payload = {"type": "token", "content": text}
                        if parent_id := meta.get("parent_tool_call_id"):
                            payload["parent_tool_call_id"] = parent_id
                        yield payload
