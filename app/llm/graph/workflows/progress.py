from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt

from app.events.context import current_event_context
from app.events.errors import EventPublicationError


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
    """Publish one start and one terminal progress event for a subagent node."""

    accepts_config = "config" in inspect.signature(node).parameters

    def payload(
        state: Mapping[str, Any],
        update: Mapping[str, Any] | None,
        *,
        phase_state: str,
        iteration: int | None,
        error: str | None = None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {"phase_state": phase_state}
        if data_builder is not None:
            data.update(dict(data_builder(state, update)))
        if error:
            data["error"] = error
        payload = {
            "node": node_name,
            "phase": phase,
            "phase_label": phase_label,
            "status": "error" if phase_state == "failed" else "running",
            "data": data,
        }
        if iteration is not None:
            payload["iteration"] = iteration
        return payload

    async def wrapped(
        state: Mapping[str, Any],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        event_context = current_event_context().scoped(node=node_name)
        iteration = (
            iteration_resolver(state)
            if iteration_resolver is not None
            else None
        )
        await event_context.bus.publish(
            event_context.event(
                "subagent_progress",
                payload(
                    state,
                    None,
                    phase_state="started",
                    iteration=iteration,
                ),
            ),
        )
        try:
            result = await (node(state, config) if accepts_config else node(state))
        except GraphInterrupt:
            raise
        except EventPublicationError:
            raise
        except Exception as exc:
            await event_context.bus.publish(
                event_context.event(
                    "subagent_progress",
                    payload(
                        state,
                        None,
                        phase_state="failed",
                        iteration=iteration,
                        error=str(exc),
                    ),
                ),
            )
            raise

        error = result.get("error")
        await event_context.bus.publish(
            event_context.event(
                "subagent_progress",
                payload(
                    state,
                    result,
                    phase_state="failed" if error else "completed",
                    iteration=iteration,
                    error=str(error) if error else None,
                ),
            ),
        )
        return result

    return wrapped
