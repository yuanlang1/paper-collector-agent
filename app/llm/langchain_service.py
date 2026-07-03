import logging
import json
from datetime import datetime
from tkinter import E

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.agents import create_agent
from langchain_core.messages.tool import tool_call
from langchain_core.runnables import configurable
from langgraph.types import Command
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any

from sqlalchemy.util import to_list
from app.llm import tool_factory
from app.llm.graph.workflow import build_agent_workflow
from app.schemas.agent import RouteDecision
from app.llm.model_factory import create_chat_model, create_structured_chat_model
from app.llm.prompt import *
from app.api.settings import *
from app.llm.tools.registry import *
from app.llm.tool_factory import *
from langgraph.checkpoint.memory import InMemorySaver
from collections.abc import AsyncIterator
from app.core.response import ServiceResponse

logger = logging.getLogger(__name__)

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


class LangChainAgentService:
    def __init__(self):
        self.ask_llm = create_chat_model(temperature=0.7)
        self.router = create_structured_chat_model(RouteDecision, temperature=0)
        self.agent_llm = create_chat_model(temperature=0.3)
        self.agent_stream_llm = create_chat_model(temperature=0.3, streaming=True)
        self.tool_factory = LangChainToolFactory()
        self.checkpointer = InMemorySaver()

        self.graph = build_agent_workflow(
            router = self.router,
            ask_llm = self.ask_llm,
            agent_llm= self.agent_llm,
            tool_factory = self.tool_factory,
            build_system_prompt = self.build_system_prompt,
            checkpointer = self.checkpointer,
        )
    
    def _build_chat_response(
        self,
        result: Dict[str, Any],
        conversation_id: str,
    ) -> Dict[str, Any]:
        interrupts = result.get("__interrupt__")

        if interrupts:
            interrupt = interrupts[0]
            interrupt_value = getattr(interrupt, "value", interrupt)

            return {
                "conversation_id": conversation_id,
                "status": "confirmation_required",
                "route": result.get("route"),
                "reply": interrupt_value.get("message", "需要人工确认后继续。")
                    if isinstance(interrupt_value, dict)
                    else "需要人工确认后继续。",
                "requires_confirmation": True,
                "reasoning": result.get("reasoning"),
                "actions": result.get("actions") or [],
                "metadata": {
                    "thread_id": conversation_id,
                    "workflow": " -> ".join(result.get("workflow") or []),
                    "interrupt": interrupt_value,
                },
            }

        actions = result.get("actions") or []
        status = result.get("status")

        if not status:
            if result.get("requires_confirmation"):
                status = "confirmation_required"
            elif actions:
                status = "tool_executed"
            else:
                status = "completed"

        return {
            "conversation_id": conversation_id,
            "status": status,
            "route": result.get("route"),
            "reply": result.get("reply") or "",
            "requires_confirmation": result.get("requires_confirmation", False),
            "reasoning": result.get("reasoning"),
            "actions": actions,
            "metadata": {
                "thread_id": conversation_id,
                "workflow": " -> ".join(result.get("workflow") or []),
                "allowed_tools": result.get("allowed_tools") or [],
                "human_decision": result.get("human_decision"),
            },
        }

    def build_system_prompt(
        self,
        base_prompt: str,
        db:Session,
    ) -> str:
        custom = get_custom_system_prompt(db)
        if custom and custom.strip():
            return f"{custom.strip()}\n\n---\n\n{base_prompt}"
        else:
            return base_prompt
    
    async def chat(
        self,
        message: str,
        conversation_id: str,
        db: Session,
    ) -> Dict[str, Any]:
        result = await self.graph.ainvoke(
            {
                "message": message,
                "conversation_id": conversation_id,
                "messages": [HumanMessage(content=message)],
                "actions": [],
                "workflow": [],
            },
            config = {
                "configurable": {
                    "thread_id": conversation_id,
                    "db": db,
                }
            },
        )

        return self._build_chat_response(
            result = result,
            conversation_id = conversation_id,
        )

    async def resume_chat(
        self,
        conversation_id: str, 
        resume_payload: Dict[str, Any],
        db: Session,
    ) -> Dict[str, Any]:
        result = await self.graph.ainvoke(
            Command(resume = resume_payload),
            config = {
                "configurable": {
                    "thread_id": conversation_id,
                    "db": db,
                }
            },
        )

        return self._build_chat_response(
            result = result, 
            conversation_id = conversation_id,
        )

    
    async def chat_stream(
        self,
        message: str,
        conversation_id: str,
        db: Session,
    ) -> AsyncIterator[str]:
        async for sse_message in self._run_stream(
            message = message,
            conversation_id = conversation_id,
            db = db,
        ): 
            yield sse_message

    async def _run_stream(
        self,
        message: str,
        conversation_id: str,
        db: Session,
    ) -> AsyncIterator[str]:
        try:
            yield _sse("meta", {
                "conversation_id": conversation_id,
                "thread_id": conversation_id,
            })

            yield _sse("phase", {
                "phase": "thinking",
                "message": "正在理解你的指令...",
            })

            config = {
                "configurable": {
                    "thread_id": conversation_id,
                    "db": db,
                }
            }

            graph_input = {
                "message": message,
                "conversation_id": conversation_id,
                "messages": [HumanMessage(content=message)],
                "actions": [],
                "workflow": [],
            }

            emitted_action_count = 0
            latest_state: Dict[str, Any] = {}

            async for stream_mode, chunk in self.graph.astream(
                graph_input, 
                config = config,
                stream_mode = ["messages", "updates"]
            ):
                if stream_mode == "messages":
                    msg_chunk, _metadata = chunk

                    if getattr(msg_chunk, "tool_call_chunks", None):
                        continue

                    text = self._content_to_stream_text(
                        getattr(msg_chunk, "content", "")
                    )

                    if text:
                        yield _sse("delta", {
                            "content": text,
                        })
                
                elif stream_mode == "updates":
                    interrupt_payload = self._extract_interrupt_payload(chunk)

                    if interrupt_payload is not None:
                        yield _sse("interrupt", {
                            "conversation_id": conversation_id,
                            "status": "confirmation_required",
                            "requires_confirmation": True,
                            "payload": interrupt_payload,
                        })
                        return

                    for node_name, update in chunk.items():
                        if not isinstance(update, dict):
                            continue
                            
                        latest_state.update(update)

                        if node_name == "route":
                            yield _sse("phase", {
                                "phase": "routed",
                                "route": update.get("route"),
                                "reasoning": update.get("reasoning"),
                            })
                        
                        elif node_name == "agent":
                            yield _sse("phase", {
                                "phase": "executing",
                                "message": "正在判断是否需要调用工具...",
                            })
                        
                        elif node_name == "tool":
                            yield _sse("phase", {
                                "phase": "tool",
                                "message": "正在执行工具...",
                            })

                        elif node_name == "final":
                            yield _sse("phase", {
                                "phase": "summarizing",
                                "message": "正在整理回复...",
                            })
                        
                        actions = update.get("actions") or []
                        
                        for action in actions[emitted_action_count:]:
                            yield _sse("action", action)

                        emitted_action_count = len(actions)
            
            yield _sse("done", {
                "conversation_id": conversation_id,
                "status": latest_state.get("status") or "completed",
                "route": latest_state.get("route"),
                "reply": latest_state.get("reply") or "",
                "requires_confirmation": latest_state.get("requires_confirmation", False),
                "reasoning": latest_state.get("reasoning"),
                "actions": latest_state.get("actions") or [],
                "metadata": {
                    "thread_id": conversation_id,
                    "workflow": " -> ".join(latest_state.get("workflow") or []),
                    "allowed_tools": latest_state.get("allowed_tools") or [],
                    "human_decision": latest_state.get("human_decision"),
                },
            })

        except Exception as e:
            logger.exception("Agent stream failed")
            yield _sse("error", {
                "conversation_id": conversation_id,
                "status": "failed",
                "message": str(e)
            })
    
    def _content_to_stream_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []

            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))

            return "".join(parts)

        return ""

    def _extract_interrupt_payload(
        self, 
        chunk: Any,
    ) -> Any:
        if not isinstance(chunk, dict):
            return None

        interrupts = chunk.get("__interrupt__")

        if interrupts is None:
            for value in chunk.values():
                if isinstance(value, dict) and "__interrupt__" in value:
                    interrupts = value["__interrupt__"]
                    break
        
        if not interrupts:
            return None

        interrupt = interrupts[0] if isinstance(interrupts, (list, tuple)) else interrupts

        return getattr(interrupt, "value", interrupt)

_agent_service: Optional[LangChainAgentService] = None

def get_agent_service() -> LangChainAgentService:
    global _agent_service

    if _agent_service is None:
        _agent_service = LangChainAgentService()

    return _agent_service