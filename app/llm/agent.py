import asyncio
import logging
from typing import Any

from app.events.adapter import event_from_envelope
from app.events.bus import EventBus
from app.events.context import EventContext, bind_event_context
from app.events.errors import EventPublicationError
from langgraph.errors import GraphInterrupt
from app.database import SessionLocal
from app.llm.artifacts.access import ArtifactAccessService
from app.llm.graph.main.native_tools import build_native_tool_schemas
from app.llm.graph.main.nodes.memory import MemoryNode
from app.llm.graph.main.nodes.solve import SolveNode
from app.llm.graph.main.workflow import build_main_agent_workflow
from app.llm.graph.workflows.paper_search.nodes.generate_queries import (
    BuildSourceQueryPlanNode,
)
from app.llm.provider import ChatClient, ModelOptions
from app.llm.tool_adapter import to_openai_tool_schemas
from app.llm.tools.registry import build_tool_registry
from app.services.llm_profile_service import LlmRuntimeConfig
from app.llm.response import build_done_payload
from app.llm.streaming import AgentStreamAdapter
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


class AgentService:
    def __init__(
        self,
        checkpointer,
        subagent_registry: SubAgentRegistry | None = None,
        llm_config: LlmRuntimeConfig | None = None,
        memory_llm_config: LlmRuntimeConfig | None = None,
        artifact_access_service: ArtifactAccessService | None = None,
    ):
        self.checkpointer = checkpointer
        self.llm_config = llm_config
        self.memory_llm_config = memory_llm_config
        self.artifact_access_service = artifact_access_service
        self.tool_registry = build_tool_registry()
        self.chat = ChatClient(llm_config)
        if subagent_registry is None:
            source_query_plan_node = BuildSourceQueryPlanNode(chat=self.chat)
            self.subagent_registry = build_default_subagent_registry(
                source_query_plan_model=source_query_plan_node.model,
                chat=self.chat,
            )
        else:
            self.subagent_registry = subagent_registry
        model = self.chat.bind_tools(
            to_openai_tool_schemas(
                build_native_tool_schemas(
                    self.tool_registry,
                    self.subagent_registry,
                )
            ),
            options=ModelOptions(temperature=0),
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
                chat=self.chat,
                tool_registry=self.tool_registry,
                subagent_registry=self.subagent_registry,
            ),
            subagent_registry=self.subagent_registry,
            tool_registry=self.tool_registry,
            artifact_access_service=self.artifact_access_service,
            checkpointer=self.checkpointer,
        )

    async def get_state(self, session: Session):
        return await self.graph.aget_state(session.graph_config)

    async def stream(
        self,
        session: Session,
        *,
        event_bus: EventBus,
    ) -> dict[str, Any]:
        adapter = AgentStreamAdapter(self.subagent_registry)
        latest_root_state: dict[str, Any] = {}

        with bind_event_context(
            EventContext(bus=event_bus, run_id=str(session.run_id)),
        ) as event_context:
            try:
                await event_bus.publish(
                    event_context.event("run_started", {"status": "running"}),
                )

                async for namespace, stream_type, chunk in self.graph.astream(
                    session.graph_input,
                    config=session.graph_config,
                    stream_mode=["messages", "updates"],
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
                            await event_bus.publish(
                                event_context.event("message", {"content": text}),
                            )
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
                        await event_bus.publish(
                            event_from_envelope(
                                adapter.confirmation_required(response),
                                run_id=str(session.run_id),
                            ),
                        )
                        break

                    if not isinstance(chunk, dict):
                        continue

                    for node_name, update in chunk.items():
                        if not isinstance(update, dict):
                            continue
                        if namespace:
                            continue

                        latest_root_state.update(update)
                        for event in adapter.handle_update(
                            node_name=node_name,
                            update=update,
                        ):
                            await event_bus.publish(
                                event_from_envelope(
                                    event,
                                    run_id=str(session.run_id),
                                ),
                            )
                else:
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
                    await event_bus.publish(event_context.event(event_name, response))

            except asyncio.CancelledError:
                raise
            except GraphInterrupt:
                raise
            except EventPublicationError:
                raise
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
                await event_bus.publish(event_context.event("run_failed", response))

        return response

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
