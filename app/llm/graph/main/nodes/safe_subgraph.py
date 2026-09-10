from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt

from app.llm.graph.main.nodes.tool import build_action_result_update
from app.llm.streaming.notify import langgraph_notifier
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


class SubAgentNode:

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

        scope = {
            "source": "subagent",
            "action_id": str(call["id"]),
            "delegation_id": str(call["id"]),
            "subagent": runtime.spec.name,
            "workflow": runtime.stream.workflow,
        }
        notify = langgraph_notifier(config).scoped(**scope)
        child_config = {
            **config,
            "metadata": {
                **dict(config.get("metadata") or {}),
                "notify_scope": scope,
            },
        }
        notify(
            "subagent_started",
            {
                "name": runtime.spec.name,
                "display_name": runtime.spec.display_name,
                "message": f"{runtime.spec.display_name} 已启动。",
                "progress": 0,
                "progress_percent": 0,
                "status": "running",
                "data": {},
            },
        )
        try:
            result = await runtime.graph.ainvoke(state, config=child_config)
        except GraphInterrupt:
            raise
        except Exception as exc:
            logger.exception("Subgraph failed: %s", runtime.spec.name)
            notify(
                "subagent_failed",
                {
                    "name": runtime.spec.name,
                    "message": str(exc),
                    "status": "error",
                    "data": {"error_code": runtime.error_code},
                },
            )
            return build_subagent_failure_update(
                call=call,
                subagent=runtime.spec.name,
                error_code=runtime.error_code,
                summary=runtime.failure_summary,
                exception=exc,
            )

        action_result = result.get("last_action_result")
        if not isinstance(action_result, Mapping):
            error = RuntimeError("subgraph returned without last_action_result")
            notify(
                "subagent_failed",
                {
                    "name": runtime.spec.name,
                    "message": str(error),
                    "status": "error",
                    "data": {"error_code": runtime.error_code},
                },
            )
            return build_subagent_failure_update(
                call=call,
                subagent=runtime.spec.name,
                error_code=runtime.error_code,
                summary=runtime.failure_summary,
                exception=error,
            )

        status = str(action_result.get("status") or "error")
        completed = status in {"success", "partial"}
        notify(
            "subagent_completed" if completed else "subagent_failed",
            {
                "name": runtime.spec.name,
                "message": str(action_result.get("summary") or runtime.failure_summary),
                "progress": 100 if completed else None,
                "progress_percent": 100 if completed else None,
                "status": status,
                "data": dict(action_result.get("data") or {}),
            },
        )
        return result
