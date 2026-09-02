from collections.abc import AsyncIterator, Awaitable, Callable
import logging
from typing import Any

from app.config import settings
from app.llm.graph.main.native_tools import build_native_tool_schemas
from app.llm.graph.main.nodes.solve import SolveNode
from app.llm.graph.main.workflow import build_main_agent_workflow
from app.llm.graph.workflows.paper_search.workflow import build_paper_search_workflow
from app.llm.graph.workflows.review_generate.workflow import build_task_review_workflow
from app.llm.model_factory import create_chat_model, use_llm_runtime_config
from app.services.llm_profile_service import LlmRuntimeConfig
from app.llm.response import build_chat_response, build_done_payload
from app.llm.streaming import AgentStreamAdapter
from app.llm.streaming.card_snapshot import CardMetaAccumulator
from app.llm.streaming.utils import (
    content_to_text,
    extract_interrupt_payload,
    to_jsonable,
)
from app.llm.subagents.registry import ALL_SUBAGENTS, SubAgentRegistry
from app.runtime.session import Session


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
    ):
        self.checkpointer = checkpointer
        self.llm_config = llm_config
        self.subagent_registry = subagent_registry or SubAgentRegistry(ALL_SUBAGENTS)
        with use_llm_runtime_config(llm_config):
            model = create_chat_model(temperature=0).bind_tools(build_native_tool_schemas())
            self.graph = build_main_agent_workflow(
                solve_node=SolveNode(model=model),
                paper_search_graph=build_paper_search_workflow(skip_confirmation=True),
                task_review_graph=build_task_review_workflow(),
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
        adapter = AgentStreamAdapter()
        card_meta = CardMetaAccumulator(
            model=(getattr(self, "llm_config", None).model if getattr(self, "llm_config", None) else settings.OPENAI_MODEL),
            provider=(getattr(self, "llm_config", None).provider if getattr(self, "llm_config", None) else getattr(settings, "LLM_PROVIDER", None)),
        )
        latest_root_state: dict[str, Any] = {}
        paper_search_tool_call_id: str | None = None
        paper_search_states: dict[str, dict[str, Any]] = {}
        task_review_tool_call_id: str | None = None
        task_review_states: dict[str, dict[str, Any]] = {}

        def emit(event: str, data: dict[str, Any]) -> str:
            envelope = session.build_sse_envelope(event, data)
            card_meta.observe(envelope)
            return session.encode_sse_envelope(envelope)

        try:
            yield emit("run_started", {"status": "running"})

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
                        yield emit("message", {"content": text})
                    continue

                if stream_type == "custom":
                    for event_name, payload in adapter.handle_custom(chunk):
                        timeline_context: dict[str, Any] = {}
                        if event_name == "timeline_step":
                            workflow = payload.get("workflow")
                            if workflow == "paper_search":
                                timeline_context = {
                                    "delegation_id": paper_search_tool_call_id,
                                    "child_thread_id": (
                                        f"{session.conversation_id}:paper_search_agent:"
                                        f"{paper_search_tool_call_id}"
                                        if paper_search_tool_call_id is not None
                                        else next(
                                            (
                                                str(item)
                                                for item in namespace
                                                if str(item).split(":", 1)[0]
                                                == "paper_search"
                                            ),
                                            None,
                                        )
                                    ),
                                }
                            elif workflow == "task_review":
                                timeline_context = {
                                    "delegation_id": task_review_tool_call_id,
                                    "child_thread_id": (
                                        f"{session.conversation_id}:task_review_agent:"
                                        f"{task_review_tool_call_id}"
                                        if task_review_tool_call_id is not None
                                        else next(
                                            (
                                                str(item)
                                                for item in namespace
                                                if str(item).split(":", 1)[0]
                                                == "task_review"
                                            ),
                                            None,
                                        )
                                    ),
                                }
                        yield emit(
                            event_name,
                            {
                                **payload,
                                **timeline_context,
                                "namespace": list(namespace),
                            },
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
                    event_name, payload = adapter.confirmation_required(
                        interrupt_payload,
                    )
                    terminal_event = emit(
                        event_name,
                        {**payload, "namespace": list(namespace)},
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
                        if node_name == "prepare_paper_search":
                            action_id = update.get("paper_search_tool_call_id")
                            if action_id is not None:
                                paper_search_tool_call_id = str(action_id)
                        if node_name == "prepare_task_review":
                            action_id = update.get("task_review_tool_call_id")
                            if action_id is not None:
                                task_review_tool_call_id = str(action_id)

                    paper_search_namespace = next(
                        (
                            str(item)
                            for item in namespace
                            if str(item).split(":", 1)[0] == "paper_search"
                        ),
                        None,
                    )
                    if paper_search_namespace is not None:
                        state = paper_search_states.setdefault(
                            paper_search_namespace,
                            {},
                        )
                        state.update(update)
                        event_name, payload = adapter.subagent_progress(
                            subagent="paper_search_agent",
                            action_id=paper_search_tool_call_id,
                            child_thread_id=(
                                f"{session.conversation_id}:paper_search_agent:"
                                f"{paper_search_tool_call_id}"
                                if paper_search_tool_call_id is not None
                                else paper_search_namespace
                            ),
                            checkpoint_namespace=paper_search_namespace,
                            node_name=node_name,
                            update=state,
                            workflow="paper_search",
                        )
                        yield emit(
                            event_name,
                            {**payload, "namespace": list(namespace)},
                        )

                    task_review_namespace = next(
                        (
                            str(item)
                            for item in namespace
                            if str(item).split(":", 1)[0] == "task_review"
                        ),
                        None,
                    )
                    if task_review_namespace is not None:
                        state = task_review_states.setdefault(
                            task_review_namespace,
                            {},
                        )
                        state.update(update)
                        event_name, payload = adapter.subagent_progress(
                            subagent="task_review_agent",
                            action_id=task_review_tool_call_id,
                            child_thread_id=(
                                f"{session.conversation_id}:task_review_agent:"
                                f"{task_review_tool_call_id}"
                                if task_review_tool_call_id is not None
                                else task_review_namespace
                            ),
                            checkpoint_namespace=task_review_namespace,
                            node_name=node_name,
                            update=state,
                            workflow="task_review",
                        )
                        yield emit(
                            event_name,
                            {**payload, "namespace": list(namespace)},
                        )

                    for event_name, payload in adapter.handle_update(
                        node_name=node_name,
                        update=update,
                    ):
                        yield emit(
                            event_name,
                            {**payload, "namespace": list(namespace)},
                        )

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
            terminal_event = emit(event_name, response)
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

            terminal_event = emit("run_failed", response)
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
