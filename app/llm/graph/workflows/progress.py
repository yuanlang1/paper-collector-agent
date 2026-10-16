from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp
from langgraph.runtime import Runtime

from app.bus.events import SubagentProgressEvent


Node = Callable[..., Awaitable[dict[str, Any]]]
ProgressDataBuilder = Callable[
    [Mapping[str, Any], Mapping[str, Any] | None],
    Mapping[str, Any],
]
IterationResolver = Callable[[Mapping[str, Any]], int | None]


def instrument_subagent_progress_node(
    *,
    node_name: str,
    node: Node,
    phase: str,
    phase_label: str,
    data_builder: ProgressDataBuilder | None = None,
    iteration_resolver: IterationResolver | None = None,
) -> Node:
    """Send progress through the graph runtime's stream writer."""
    parameters = inspect.signature(node).parameters

    async def wrapped(
        state: Mapping[str, Any],
        config: RunnableConfig,
        runtime: Runtime,
    ) -> dict[str, Any]:
        context = runtime.context

        def emit(description: str) -> None:
            runtime.stream_writer(SubagentProgressEvent(
                **context["event_fields"], ts=datetime.now(UTC).isoformat(), description=description,
            ))

        iteration = iteration_resolver(state) if iteration_resolver else None
        label = phase_label + (f"（第 {iteration} 轮）" if iteration else "")
        emit(f"开始：{label}。")
        kwargs = {}
        if "config" in parameters:
            kwargs["config"] = config
        if "runtime" in parameters:
            kwargs["runtime"] = runtime
        try:
            result = await node(state, **kwargs)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            emit(f"失败：{label}。{exc}")
            raise

        error = result.get("error")
        description = f"失败：{label}。{error}" if error else f"完成：{label}。"
        if data_builder:
            counts = data_builder(state, result).get("counts")
            if counts:
                description += " " + "，".join(f"{key}={value}" for key, value in counts.items())
        emit(description)
        return result

    return wrapped
