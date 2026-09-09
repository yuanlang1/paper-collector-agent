from collections.abc import AsyncIterator, Awaitable, Callable
import logging
from typing import Any

from app.config import settings
from app.database import SessionLocal
from app.llm.graph.main.native_tools import build_native_tool_schemas
from app.llm.graph.main.nodes.memory import MemoryNode
from app.llm.graph.main.nodes.solve import SolveNode
from app.llm.graph.main.workflow import build_main_agent_workflow
from app.llm.graph.workflows.paper_search.nodes.generate_queries import (
    BuildSourceQueryPlanNode,
)
from app.llm.model_factory import create_chat_model, use_llm_runtime_config
from app.llm.tool_adapter import to_openai_tool_schemas
from app.llm.tools.registry import build_tool_registry
from app.services.llm_profile_service import LlmRuntimeConfig
from app.llm.response import build_chat_response, build_done_payload
from app.llm.streaming import AgentStreamAdapter
from app.llm.streaming.card_snapshot import CardMetaAccumulator
from app.llm.streaming.notify import EventEnvelope, build_event
from app.llm.streaming.utils import (
    content_to_text,
    extract_interrupt_payload,
    to_jsonable,
)
from app.llm.subagents.registry import (
    SubAgentRegistry,
    build_default_subagent_registry,
)
from app.runtime.session import Session
from app.runtime.system_context import SystemContextBuilder


logger = logging.getLogger(__name__)


TerminalCallback = Callable[
    [dict[str, Any], dict[str, Any]],
    Awaitable[None],
]


class AgentService:
    def __init__(
        self,
        checkpointer,
        subagent_registry: SubAgentRegistry | None = None,
        llm_config: LlmRuntimeConfig | None = None,
        memory_llm_config: LlmRuntimeConfig | None = None,
    ):
        self.checkpointer = checkpointer
        self.llm_config = llm_config
        self.memory_llm_config = memory_llm_config
        self.tool_registry = build_tool_registry()
        with use_llm_runtime_config(llm_config):
            if subagent_registry is None:
                source_query_plan_node = BuildSourceQueryPlanNode()
                self.subagent_registry = build_default_subagent_registry(
                    source_query_plan_model=source_query_plan_node.model,
                )
            else:
                self.subagent_registry = subagent_registry
            model = create_chat_model(temperature=0).bind_tools(
                to_openai_tool_schemas(
                    build_native_tool_schemas(
                        self.tool_registry,
                        self.subagent_registry,
                    )
                )
            )
            self.graph = build_main_agent_workflow(
                memory_node=MemoryNode(
                    db_factory=SessionLocal,
                    system_context_builder=SystemContextBuilder(),
                    llm_config=llm_config,
                    memory_llm_config=memory_llm_config,
                ),
                solve_node=SolveNode(
                    model=model,
                    tool_registry=self.tool_registry,
                    subagent_registry=self.subagent_registry,
                ),
                subagent_registry=self.subagent_registry,
                tool_registry=self.tool_registry,
                checkpointer=self.checkpointer,
            )

    async def get_state(self, session: Session):
        return await self.graph.aget_state(session.graph_config)

    async def invoke(self, session: Session) -> dict[str, Any]:
        result = await self.graph.ainvoke(
            session.graph_input,
            config=session.graph_config,
        )
        return build_chat_response(
            result=result,
            conversation_id=session.conversation_id,
            run_id=str(session.run_id),
        )

    async def stream(
        self,
        session: Session,
        *,
        on_terminal: TerminalCallback | None = None,
    ) -> AsyncIterator[str]:
        adapter = AgentStreamAdapter(self.subagent_registry)
        card_meta = CardMetaAccumulator(
            model=(getattr(self, "llm_config", None).model if getattr(self, "llm_config", None) else settings.OPENAI_MODEL),
            provider=(getattr(self, "llm_config", None).provider if getattr(self, "llm_config", None) else getattr(settings, "LLM_PROVIDER", None)),
        )
        latest_root_state: dict[str, Any] = {}

        def encode_sse_event(event: EventEnvelope) -> str:
            envelope = session.build_sse_envelope(
                str(event["event"]),
                {key: value for key, value in event.items() if key != "event"},
            )
            card_meta.observe(envelope)
            return session.encode_sse_envelope(envelope)

        try:
            yield encode_sse_event(build_event("run_started", {"status": "running"}))

            async for namespace, stream_type, chunk in self.graph.astream(
                session.graph_input,
                config=session.graph_config,
                stream_mode=["messages", "updates", "custom"],
                subgraphs=True,
            ):
                namespace = tuple(namespace or ())

                if stream_type == "messages":
                    message_chunk, metadata = chunk
                    if namespace or metadata.get("langgraph_node") != "final":
                        continue

                    text = content_to_text(
                        getattr(message_chunk, "content", ""),
                    )
                    if text:
                        yield encode_sse_event(build_event("message", {"content": text}))
                    continue

                if stream_type == "custom":
                    for event in adapter.handle_custom(chunk):
                        yield encode_sse_event(event)
                    continue

                if stream_type != "updates":
                    continue

                interrupt_payload = extract_interrupt_payload(chunk)
                if interrupt_payload is not None:
                    response = build_done_payload(
                        state={
                            **latest_root_state,
                            "run_id": session.run_id,
                            "__interrupt__": [interrupt_payload],
                        },
                        conversation_id=session.conversation_id,
                        run_id=str(session.run_id),
                    )
                    terminal_event = encode_sse_event(
                        adapter.confirmation_required(interrupt_payload),
                    )
                    await self._notify_terminal(
                        on_terminal,
                        response,
                        card_meta.snapshot(response),
                    )
                    yield terminal_event
                    return

                if not isinstance(chunk, dict):
                    continue

                for node_name, update in chunk.items():
                    if not isinstance(update, dict):
                        continue

                    if not namespace:
                        latest_root_state.update(update)

                    for event in adapter.handle_update(
                        node_name=node_name,
                        update=update,
                    ):
                        yield encode_sse_event(event)

            final_state = await self._read_final_state(
                config=session.graph_config,
                fallback=latest_root_state,
            )
            response = build_done_payload(
                state=final_state,
                conversation_id=session.conversation_id,
                run_id=str(session.run_id),
            )
            event_name = (
                "run_failed"
                if response["status"] == "failed"
                else "run_completed"
            )
            terminal_event = encode_sse_event(build_event(event_name, response))
            await self._notify_terminal(
                on_terminal,
                response,
                card_meta.snapshot(response),
            )
            yield terminal_event

        except Exception as exc:
            logger.exception(
                "Agent stream failed: conversation_id=%s, run_id=%s",
                session.conversation_id,
                session.run_id,
            )

            response = {
                "conversation_id": session.conversation_id,
                "run_id": str(session.run_id),
                "status": "failed",
                "reply": "本次任务执行失败，请稍后重试。",
                "pending_action": None,
                "last_action_result": None,
                "artifact_refs": [],
                "interrupt": None,
                "error": str(exc),
            }

            terminal_event = encode_sse_event(build_event("run_failed", response))
            await self._notify_terminal(
                on_terminal,
                response,
                card_meta.snapshot(response),
            )
            yield terminal_event

    async def _notify_terminal(
        self,
        callback: TerminalCallback | None,
        payload: dict[str, Any],
        card_meta: dict[str, Any],
    ) -> None:
        if callback is None:
            return

        try:
            await callback(payload, card_meta)
        except Exception:
            logger.exception(
                "Terminal callback failed: conversation_id=%s, run_id=%s",
                payload.get("conversation_id"),
                payload.get("run_id"),
            )

    async def _read_final_state(
        self,
        *,
        config: dict[str, Any],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            snapshot = await self.graph.aget_state(config)
            values = to_jsonable(snapshot.values)
            if isinstance(values, dict):
                return values
        except Exception:
            logger.warning(
                "Failed to read final graph state; using streamed updates.",
                exc_info=True,
            )
        return fallback
