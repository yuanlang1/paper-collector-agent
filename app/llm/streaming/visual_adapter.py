from __future__ import annotations

from typing import Any

from app.llm.subagents.registry import SubAgentRegistry, SubAgentRuntime
from app.llm.streaming.utils import (
    to_jsonable,
)


StreamEvent = tuple[
    str,
    dict[str, Any],
]

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
                    (
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
                    (
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
                (
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

        event_name = normalized.get("event")

        if not event_name:
            return [(
                    "progress",
                    normalized,
                )
            ]

        data = {
            key: value
            for key, value in normalized.items()
            if key != "event"
        }

        return [(
                str(event_name),
                data,
            )
        ]

    def confirmation_required(
        self,
        payload: Any,
    ) -> StreamEvent:
        return (
            "confirmation_required",
            {
                "interrupt": to_jsonable(payload)
            },
        )

    def subagent_progress(
        self,
        *,
        runtime: SubAgentRuntime,
        action_id: str | None,
        child_thread_id: str,
        checkpoint_namespace: str,
        node_name: str,
        update: dict[str, Any],
    ) -> StreamEvent:
        stream = runtime.stream
        phases = stream.phases
        phase_key = stream.node_phases.get(node_name, phases[0][0])
        phase_index = next(
            (
                index
                for index, (key, _label) in enumerate(
                    phases,
                    start=1,
                )
                if key == phase_key
            ),
            1,
        )
        phase_label = dict(phases)[phase_key]
        stage = update.get("stage")
        terminal = node_name in stream.terminal_nodes
        iteration = None
        if stream.iteration_key:
            try:
                iteration = int(update.get(stream.iteration_key, 0)) + 1
            except (TypeError, ValueError):
                iteration = 1
        task_id = next(
            (
                update.get(key)
                for key in stream.task_id_keys
                if update.get(key) is not None
            ),
            None,
        )
        progress_percent = (
            100
            if terminal
            else round(
                phase_index / len(phases) * 100
            )
        )

        return (
            "subagent_progress",
            {
                "subagent": runtime.spec.name,
                "workflow": stream.workflow,
                "delegation_id": action_id,
                "child_thread_id": child_thread_id,
                "checkpoint_namespace": checkpoint_namespace,
                "node": node_name,
                "phase": phase_key,
                "phase_label": phase_label,
                "phase_index": phase_index,
                "phase_count": len(phases),
                "progress_percent": progress_percent,
                "terminal": terminal,
                "task_id": task_id,
                "stage": stage,
                "status": update.get("status"),
                "progress": to_jsonable(update.get("progress") or {}),
                "warnings": to_jsonable(update.get("warnings") or []),
                "error": update.get("error"),
                "task_status_update_error": update.get(
                    "task_status_update_error"
                ),
                "pdf_cleanup_error": update.get(
                    "pdf_cleanup_error"
                ),
                "degraded": bool(update.get("degraded")),
                "remote_task_state": update.get(
                    "remote_task_state"
                ),
                "supplemental_search_round": update.get(
                    "supplemental_search_round",
                    0,
                ),
                "iteration": iteration,
                "source_stats": to_jsonable(
                    update.get("source_search_stats") or {}
                ),
            },
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
