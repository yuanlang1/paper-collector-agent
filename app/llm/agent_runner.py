from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from app.common.exceptions import BadRequestException
from app.llm.loop.agent_loop import AgentLoop
from app.llm.tools.registry import ToolRegistry


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class AgentRunContext:
    session_id: str
    run_id: str
    user_id: str


class AgentRunner:
    def __init__(
        self,
        *,
        thread_id: str,
        run_id: str,
        user_id: str,
        model: BaseChatModel,
        tool_registry: ToolRegistry,
        checkpointer,
        store,
    ) -> None:
        self.context = AgentRunContext(
            session_id=thread_id,
            run_id=run_id,
            user_id=user_id,
        )
        self.checkpointer = checkpointer
        self.store = store
        self.model = model
        self.tool_registry = tool_registry
        self.config = {
            "configurable": {"thread_id": thread_id},
            "metadata": {"user_id": user_id, "app_run_id": run_id},
        }

    def _event_fields(self) -> dict[str, str]:
        return {
            "run_id": self.context.run_id,
        }

    def _tool_output_preview(self, output: Any, limit: int = 200) -> str:
        if isinstance(output, ToolMessage):
            output = output.content
        text = output if isinstance(output, str) else str(output)

        return text if len(text) <= limit else text[:limit] + "…"

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
            model=self.model,
            tools=self.tool_registry.get_tools(),
            system_prompt="你是一个助手",
            checkpointer=self.checkpointer,
            store=self.store,
            context_schema=AgentRunContext,
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

        yield {"type": "status", "message": "正在思考…"}

        async with aclosing(
            self.agent.stream(
                messages=messages,
                config=self.config,
                context=self.context,
            )
        ) as stream:
            async for event in stream:
                kind = event.get("event")
                if kind == "on_tool_start":
                    name = event.get("name") or ""
                    yield {
                        "type": "tool_start",
                        "name": name,
                        "label": self.tool_registry.get_preview_name(name),
                    }
                elif kind == "on_tool_end":
                    name = event.get("name") or ""
                    yield {
                        "type": "tool_end",
                        "name": name,
                        "label": self.tool_registry.get_preview_name(name),
                        "preview": self._tool_output_preview(
                            (event.get("data") or {}).get("output")
                        ),
                    }
                elif kind == "on_chat_model_stream":
                    meta = event.get("metadata") or {}
                    if meta.get("langgraph_node") not in (None, "solve"):
                        continue
                    text = self._chunk_text((event.get("data") or {}).get("chunk"))
                    if text:
                        yield {"type": "token", "content": text}
