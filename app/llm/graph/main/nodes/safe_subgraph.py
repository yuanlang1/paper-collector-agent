from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt


logger = logging.getLogger(__name__)


def build_subagent_error_handoff(
    *,
    subagent: str,
    error_code: str,
    summary: str,
    exception: Exception,
    retryable: bool = False,
) -> dict[str, Any]:
    return {
        "status": "error",
        "summary": summary,
        "data": {
            "subagent": subagent,
            "exception_type": type(exception).__name__,
        },
        "artifact_refs": [],
        "retryable": retryable,
        "error_code": error_code,
        "error_message": str(exception),
    }


class SafeSubgraphNode:
    """Convert a subgraph exception into the handoff consumed by its parent."""

    def __init__(
        self,
        *,
        subgraph: Any,
        handoff_key: str,
        subagent: str,
        error_code: str,
        summary: str,
    ) -> None:
        self.subgraph = subgraph
        self.handoff_key = handoff_key
        self.subagent = subagent
        self.error_code = error_code
        self.summary = summary

    async def __call__(
        self,
        state: Mapping[str, Any],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        try:
            return await self.subgraph.ainvoke(state, config=config)
        except GraphInterrupt:
            raise
        except Exception as exc:
            logger.exception("Subgraph failed: %s", self.subagent)
            return {
                self.handoff_key: build_subagent_error_handoff(
                    subagent=self.subagent,
                    error_code=self.error_code,
                    summary=self.summary,
                    exception=exc,
                )
            }
