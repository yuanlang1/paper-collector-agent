from collections.abc import (
    AsyncIterator,
)
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any, Optional
from uuid import uuid4

from langchain_core.messages import (
    HumanMessage,
)
from langgraph.checkpoint.memory import (
    InMemorySaver,
)
from langgraph.types import Command
from sqlalchemy.orm import Session

from app.llm.graph.main.nodes.solve import (
    SolveNode,
)
from app.llm.graph.main.schemas import (
    RunStatus,
    SolveDecision,
)
from app.llm.graph.main.workflow import (
    build_main_agent_workflow,
)
from app.llm.graph.workflows.paper_search.workflow import (
    build_paper_search_workflow,
)
from app.llm.subagents.paper_search.agent import (
    build_paper_search_agent_spec,
)
from app.llm.subagents.registry import (
    SubAgentRegistry,
)
from app.llm.model_factory import (
    create_structured_chat_model,
)
from app.llm.response import (
    build_chat_response,
    build_done_payload,
)
from app.llm.streaming import (
    AgentStreamAdapter,
)
from app.llm.streaming.utils import (
    content_to_text,
    encode_sse,
    extract_interrupt_payload,
    to_jsonable,
)


logger = logging.getLogger(
    __name__
)


@dataclass(frozen=True)
class _GraphRun:
    run_id: str
    graph_input: Any
    config: dict[str, Any]


@dataclass
class _StreamSession:
    conversation_id: str
    run_id: str
    sequence: int = 0

    def encode(
        self,
        event: str,
        data: dict[str, Any],
    ) -> str:
        self.sequence += 1

        return encode_sse(
            event,
            {
                "event_id": f"{self.run_id}:{self.sequence}",
                "sequence": self.sequence,
                "event": event,
                "conversation_id": self.conversation_id,
                "run_id": self.run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": data,
            },
        )


def _string_value(
    value: Any,
) -> str | None:
    if value is None:
        return None

    if hasattr(value, "value"):
        return str(value.value)

    return str(value)


class AgentService:
    def __init__(
        self,
        subagent_registry: SubAgentRegistry | None = None
    ):
        self.checkpointer = InMemorySaver()
        self.subagent_registry = subagent_registry or SubAgentRegistry(
            [build_paper_search_agent_spec()]
        )
        solver = create_structured_chat_model(SolveDecision, temperature = 0)
        solve_node = SolveNode(solver = solver, subagent_registry = self.subagent_registry)
        paper_search_graph = build_paper_search_workflow(
            skip_confirmation=True,
        )

        self.graph = (
            build_main_agent_workflow(
                solve_node = solve_node,
                paper_search_graph = paper_search_graph,
                checkpointer = self.checkpointer,
            )
        )

    def _build_config(
        self,
        *,
        conversation_id: str,
        db: Session,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        configurable: dict[str, Any] = {
            "thread_id": conversation_id,
            "db": db,
        }

        if run_id:
            configurable["run_id"] = run_id

        metadata: dict[str, Any] = {
            "conversation_id": conversation_id,
            "graph": "solve_tool_v1",
        }

        if run_id:
            metadata["run_id"] = run_id

        return {
            "configurable": configurable,
            "metadata": metadata,
            "recursion_limit": 50,
        }

    def _prepare_run(
        self,
        *,
        message: str,
        conversation_id: str,
        db: Session,
        forced_subagent: dict[str, Any] | None = None,
    ) -> _GraphRun:
        run_id = f"run_{uuid4().hex}"

        graph_input = {
            "conversation_id": conversation_id,
            "run_id": run_id,
            "forced_subagent": forced_subagent,
            "paper_search_request": None,
            "paper_search_handoff": None,
            "paper_search_action_id": None,
            "messages": [HumanMessage(content = message)],
            "solve_decision": None,
            "pending_action": None,
            "confirmation": None,
            "last_action_result": None,
            "action_history": [],
            "artifact_refs": [],
            "reply": "",
            "run_status": "running",
            "error": None,
            "action_rounds": 0,
            "max_action_rounds": 8,
        }

        return _GraphRun(
            run_id = run_id,
            graph_input = graph_input,
            config = self._build_config(
                conversation_id = conversation_id,
                run_id = run_id,
                db = db,
            ),
        )

    async def _prepare_resume(
        self,
        *,
        conversation_id: str,
        resume_payload: dict[str, Any],
        db: Session,
    ) -> _GraphRun:
        config = self._build_config(
            conversation_id = conversation_id,
            db = db,
        )

        snapshot = await self.graph.aget_state(config)

        values = snapshot.values or {}

        if not values:
            raise ValueError("没有找到可恢复的运行")

        if not snapshot.next:
            raise ValueError("当前运行没有等待恢复")

        run_id = values.get("run_id")

        if not run_id:
            raise ValueError("待恢复状态缺少 run_id")

        run_id = str(run_id)

        return _GraphRun(
            run_id = run_id,
            graph_input = Command(resume = resume_payload),
            config = self._build_config(
                conversation_id = conversation_id,
                run_id =  run_id,
                db = db,
            ),
        )

    async def chat(
        self,
        message: str,
        conversation_id: str,
        db: Session,
        forced_subagent: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run = self._prepare_run(
            message = message,
            conversation_id = conversation_id,
            db = db,
            forced_subagent = forced_subagent,
        )

        result = await self.graph.ainvoke(
            run.graph_input,
            config = run.config,
        )

        return build_chat_response(
            result = result,
            conversation_id = conversation_id,
            run_id = run.run_id,
        )

    async def resume_chat(
        self,
        conversation_id: str,
        resume_payload: dict[str, Any],
        db: Session,
    ) -> dict[str, Any]:
        run = await self._prepare_resume(
            conversation_id = conversation_id,
            resume_payload = resume_payload,
            db = db,
        )

        result = await self.graph.ainvoke(
            run.graph_input,
            config=run.config,
        )

        return build_chat_response(
            result=result,
            conversation_id=conversation_id,
            run_id=run.run_id,
        )

    async def chat_stream(
        self,
        message: str,
        conversation_id: str,
        db: Session,
        forced_subagent: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        run = self._prepare_run(
            message = message,
            conversation_id = conversation_id,
            db = db,
            forced_subagent = forced_subagent,
        )

        async for event in (
            self._stream_graph(
                run = run,
                conversation_id = conversation_id,
            )
        ):
            yield event

    async def resume_chat_stream(
        self,
        conversation_id: str,
        resume_payload: dict[str, Any],
        db: Session,
    ) -> AsyncIterator[str]:
        run = await self._prepare_resume(
            conversation_id = conversation_id,
            resume_payload = resume_payload,
            db = db,
        )

        async for event in (
            self._stream_graph(
                run = run,
                conversation_id = conversation_id,
            )
        ):
            yield event

    async def _stream_graph(
        self,
        *,
        run: _GraphRun,
        conversation_id: str,
    ) -> AsyncIterator[str]:
        session = _StreamSession(
            conversation_id = conversation_id,
            run_id = run.run_id,
        )
        adapter = AgentStreamAdapter()

        latest_root_state: dict[str, Any] = {}
        paper_search_action_id: str | None = None
        paper_search_states: dict[str, dict[str, Any]] = {}

        try:
            yield session.encode(
                "run_started",
                {
                    "status": "running",
                },
            )

            async for (namespace, stream_type, chunk) in self.graph.astream(
                run.graph_input,
                config = run.config,
                stream_mode = [
                    "messages",
                    "updates",
                    "custom",
                ],
                subgraphs = True,
            ):
                namespace = tuple(namespace or ())

                if stream_type == "messages":
                    (message_chunk, metadata) = chunk

                    if namespace:
                        continue

                    if (metadata.get("langgraph_node") != "final"):
                        continue

                    text = content_to_text(getattr(message_chunk, "content", ""))

                    if text:
                        yield session.encode(
                            "message",
                            {
                                "content": text,
                            },
                        )

                    continue

                if stream_type == "custom":
                    for (event_name, payload) in adapter.handle_custom(chunk):
                        yield session.encode(
                            event_name,
                            {
                                **payload,
                                "namespace": list(namespace),
                            },
                        )

                    continue

                if stream_type != "updates":
                    continue

                interrupt_payload = extract_interrupt_payload(chunk)

                if (interrupt_payload is not None):
                    event_name, payload = adapter.confirmation_required(interrupt_payload)
                    
                    yield session.encode(
                        event_name,
                        {
                            **payload,
                            "namespace": list(
                                namespace
                            ),
                        },
                    )
                    return

                if not isinstance(chunk, dict):
                    continue

                for (node_name, update) in chunk.items():
                    if not isinstance(update, dict):
                        continue

                    if not namespace:
                        latest_root_state.update(update)
                        if node_name == "prepare_paper_search":
                            action_id = update.get(
                                "paper_search_action_id"
                            )
                            if action_id is not None:
                                paper_search_action_id = str(action_id)

                    paper_search_namespace = next(
                        (
                            str(item)
                            for item in namespace
                            if str(item).split(":", 1)[0]
                            == "paper_search"
                        ),
                        None,
                    )
                    if paper_search_namespace is not None:
                        paper_search_state = paper_search_states.setdefault(
                            paper_search_namespace,
                            {},
                        )
                        paper_search_state.update(update)
                        event_name, payload = adapter.subagent_progress(
                            subagent="paper_search_agent",
                            action_id=paper_search_action_id,
                            child_thread_id=(
                                f"{conversation_id}:paper_search_agent:"
                                f"{paper_search_action_id}"
                                if paper_search_action_id is not None
                                else paper_search_namespace
                            ),
                            checkpoint_namespace=paper_search_namespace,
                            node_name=node_name,
                            update=paper_search_state,
                        )
                        yield session.encode(
                            event_name,
                            {
                                **payload,
                                "namespace": list(namespace),
                            },
                        )

                    for (event_name, payload) in adapter.handle_update(
                        node_name = node_name,
                        update = update,
                    ):
                        yield session.encode(
                            event_name,
                            {
                                **payload,
                                "namespace": list(namespace),
                            },
                        )

            final_state =  await self._read_final_state(
                    config = run.config,
                    fallback = latest_root_state
                )

            response = build_done_payload(
                state = final_state,
                conversation_id = conversation_id,
                run_id = run.run_id,
            )

            if (_string_value(final_state.get("run_status")) == "failed"):
                yield session.encode("run_failed", response,)
            else:
                yield session.encode("run_completed", response)

        except Exception as exc:
            logger.exception("Agent stream failed: conversation_id=%s, run_id=%s", conversation_id, run.run_id,)

            yield session.encode(
                "run_failed",
                {
                    "status": "failed",
                    "error": str(exc),
                },
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
                "读取最终 Graph State 失败, 使用流式 update 累积结果。",
                exc_info = True,
            )

        return fallback


_agent_service: Optional[AgentService] = None


def get_agent_service() -> AgentService:
    global _agent_service

    if _agent_service is None:
        _agent_service = AgentService()

    return _agent_service
