from __future__ import annotations

from typing import Any

from app.llm.streaming.notify import EventEnvelope, build_event
from app.llm.subagents.registry import SubAgentRegistry
from app.llm.streaming.utils import (
    to_jsonable,
)


StreamEvent = EventEnvelope

class AgentStreamAdapter:
    def __init__(
        self,
        subagent_registry: SubAgentRegistry | None = None,
    ):
        self.subagent_registry = subagent_registry
        self._seen_action_starts: set[str] = set()
        self._seen_action_results: set[str] = set()

    def handle_update(
        self,
        *,
        node_name: str,
        update: dict[str, Any],
    ) -> list[StreamEvent]:
        events: list[StreamEvent] = []

        active_call = to_jsonable(update.get("active_tool_call"))
        if isinstance(active_call, dict):
            action_id = active_call.get("id")
            action_name = active_call.get("name")

            if (
                action_id
                and action_name
                and str(action_id) not in self._seen_action_starts
            ):
                self._seen_action_starts.add(str(action_id))
                events.append(
                    build_event(
                        "action_started",
                        {
                            "action_id": str(action_id),
                            "action_type": str(
                                active_call.get("kind") or "tool"
                            ),
                            "name": str(action_name),
                            "input": to_jsonable(
                                active_call.get("args") or {}
                            ),
                            "requires_confirmation": bool(
                                active_call.get(
                                    "requires_confirmation"
                                )
                            ),
                            "workflow": self._workflow_for(action_name),
                        },
                    )
                )

        result = to_jsonable(update.get("last_action_result"))

        if isinstance(result, dict):
            action_id = result.get("action_id")

            if (action_id and str(action_id) not in self._seen_action_results):
                self._seen_action_results.add(str(action_id))

                events.append(
                    build_event(
                        "action_result",
                        self._result_payload(
                            result=result,
                            node_name=node_name,
                        ),
                    )
                )

        run_status = self._string_value(update.get("run_status"))

        if (run_status == "failed" or update.get("error")):
            events.append(
                build_event(
                    "node_error",
                    {
                        "node": node_name,
                        "error": update.get("error") or "节点执行失败"
                    },
                )
            )

        return events

    def handle_custom(
        self,
        payload: Any,
    ) -> list[StreamEvent]:
        normalized = to_jsonable(payload)

        if not isinstance(normalized, dict):
            return []

        return [normalized] if normalized.get("event") else []

    def confirmation_required(
        self,
        payload: Any,
    ) -> StreamEvent:
        return build_event(
            "confirmation_required",
            {"interrupt": to_jsonable(payload)},
        )

    def _workflow_for(self, action_name: Any) -> str | None:
        if self.subagent_registry is None:
            return None
        runtime = self.subagent_registry.get_runtime(str(action_name or ""))
        return runtime.stream.workflow if runtime else None

    @staticmethod
    def _result_payload(
        *,
        result: dict[str, Any],
        node_name: str,
    ) -> dict[str, Any]:
        artifact_refs = result.get("artifact_refs")

        return {
            "node": node_name,
            "action_id": result.get("action_id"),
            "action_type": result.get("action_type"),
            "name": result.get("name"),
            "status": result.get("status"),
            "summary": result.get("summary"),
            "artifact_refs": (
                artifact_refs
                if isinstance(artifact_refs, list)
                else []
            ),
            "retryable": bool(result.get("retryable")),
            "error_code": result.get("error_code"),
            "error_message": result.get("error_message"),
        }

    @staticmethod
    def _string_value(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        if hasattr(value, "value"):
            return str(value.value)

        return str(value)
