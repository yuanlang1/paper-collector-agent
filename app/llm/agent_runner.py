
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
from typing import Any, AsyncIterator

from langchain_core.messages import HumanMessage, ToolMessage
from sqlalchemy.ext.asyncio import AsyncSession
from app.llm.loop.agent_loop import AgentLoop
from app.llm.provider import ChatClient, ModelOptions
from app.llm.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()

@dataclass
class AgentRunContext:
    session_id: str
    run_id: str
    user_id: str
    max_attempts: int

class AgentRunner:
    def __init__(
        self,
        *,
        session_id: str,
        run_id: str,
        user_id: str,
        checkpointer,
        store,
    ) -> None:
        self.context = AgentRunContext(
            session_id = session_id,
            run_id = run_id,
            user_id = user_id,
            max_attempts = 5
        )
        self.checkpointer = checkpointer
        self.store = store
        self.tool_registry = ToolRegistry()

    def _build_tools(self):
        from app.llm.subagents.paper_search import paper_search_agent
        from app.llm.subagents.task_indexing import task_indexing_agent
        from app.llm.subagents.task_review import task_review_agent

        self.tool_registry.register(paper_search_agent, "论文检索子代理")
        self.tool_registry.register(task_indexing_agent, "索引子代理")
        self.tool_registry.register(task_review_agent, "综述生成子代理")

        return self.tool_registry.get_tools()

    def _event_fields(self) -> dict[str, str]:
        return {
            "session_id": self.context.session_id,
            "run_id": self.context.run_id,
            "ts": _now(),
        }
    
    def _tool_output_preview(
        self,
        output: Any, 
        limit: int = 200
    ) -> str:
        if isinstance(output, ToolMessage):
            output = output.content
        text = output if isinstance(output, str) else str(output)

        return text if len(text) <= limit else text[:limit] + "…"

    def _chunk_text(
        self,
        chunk: Any
    ) -> str:
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

    async def run(
        self,
        db: AsyncSession,
        message: str
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            agent = AgentLoop(
                db = db, 
                model = ChatClient().model(ModelOptions(streaming = True)),
                tools = self._build_tools(),
                system_prompt = "你是一个助手",
                checkpointer = self.checkpointer,
                store = self.store,
                context_schema = AgentRunContext,
                name = "agent",
            )

            config = {
                "configurable": {
                    "thread_id": self.context.session_id,
                },
            }

            yield {"type": "status", "message": "正在思考…"}

            async for event in agent.stream(
                messages = [HumanMessage(content = message)],
                config = config,
                context = self.context,
            ):
                

                kind = event.get("event")

                if kind == "on_tool_start":
                    name = event.get("name") or ""
                    yield {
                        "type": "tool_start",
                        "name": name,
                        "label": self.tool_registry.get_preview_name(name)
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

            yield {"type": "done"}

        except RuntimeError as exc:
            logger.exception(
                "Agent run failed: session_id=%s run_id=%s",
                self.context.session_id,
                self.context.run_id,
            )

            yield {"type": "error", "message": "发生错误"}

        except Exception:
            yield {"type": "error", "message": "大模型调用失败，请稍后重试"}