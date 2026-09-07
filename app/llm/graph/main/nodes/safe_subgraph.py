from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt

from app.llm.graph.action_result import build_action_result_update
from app.llm.subagents.registry import SubAgentRegistry

logger = logging.getLogger(__name__)


def build_subagent_failure_update(
    *,
    call: Mapping[str, Any],
    subagent: str,
    error_code: str,
    summary: str,
    exception: Exception,
    retryable: bool = False,
) -> dict[str, Any]:
    return build_action_result_update(
        call=call,
        status="error",
        summary=summary,
        data={
            "subagent": subagent,
            "exception_type": type(exception).__name__,
        },
        artifact_refs=[],
        retryable=retryable,
        error_code=error_code,
        error_message=str(exception),
    )


class SafeSubgraphNode:

    def __init__(
        self,
        *,
        subagent_registry: SubAgentRegistry,
    ) -> None:
        self.subagent_registry = subagent_registry

    async def __call__(
        self,
        state: Mapping[str, Any],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        call = state.get("active_tool_call")
        if not isinstance(call, Mapping):
            raise RuntimeError("subgraph invoked without an active tool call")

        runtime = self.subagent_registry.get_runtime(
            str(call.get("name") or ""),
        )
        if runtime is None:
            return build_action_result_update(
                call=call,
                status="error",
                summary="未注册的子代理。",
                error_code="UNKNOWN_SUBAGENT",
            )

        try:
            return await runtime.graph.ainvoke(state, config=config)
        except GraphInterrupt:
            raise
        except Exception as exc:
            logger.exception("Subgraph failed: %s", runtime.spec.name)
            return build_subagent_failure_update(
                call=call,
                subagent=runtime.spec.name,
                error_code=runtime.error_code,
                summary=runtime.failure_summary,
                exception=exc,
            )
