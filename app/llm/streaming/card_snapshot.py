from __future__ import annotations

from time import perf_counter
from typing import Any

from app.llm.streaming.utils import to_jsonable


class CardMetaAccumulator:
    """Collect the user-visible execution state for one assistant card."""

    def __init__(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        self._started_at = perf_counter()
        self._model = model
        self._provider = provider
        self._iterations = 0
        self._reasoning: dict[str, dict[str, Any]] = {}
        self._tools: dict[str, dict[str, Any]] = {}
        self._subagents: dict[str, dict[str, Any]] = {}
        self._memory: dict[str, Any] | None = None

    def observe(self, envelope: dict[str, Any]) -> None:
        event = str(envelope.get("event") or "")
        data = to_jsonable(envelope.get("data") or {})
        if not isinstance(data, dict):
            return

        sequence = int(envelope.get("sequence") or 0)
        timestamp = str(envelope.get("timestamp") or "")

        if event == "iteration_started":
            self._iterations = max(
                self._iterations,
                int(data.get("iteration") or 0),
            )
        elif event == "reasoning_delta":
            self._observe_reasoning(data, sequence, timestamp)
        elif event == "action_started":
            self._observe_action_started(data, sequence, timestamp)
        elif event == "action_result":
            self._observe_action_result(data, sequence, timestamp)
        elif event in {
            "subagent_started",
            "subagent_progress",
            "subagent_completed",
            "subagent_failed",
        }:
            self._observe_subagent_event(event, data, sequence, timestamp)
        elif event == "timeline_step":
            self._observe_timeline_step(data, sequence, timestamp)
        elif event.startswith("memory_retrieval_"):
            self._observe_memory(event, data, sequence, timestamp)

    def snapshot(self, response: dict[str, Any]) -> dict[str, Any]:
        reasoning = [
            self._public_reasoning(item)
            for item in self._reasoning.values()
        ]
        if not reasoning and response.get("reasoning_content"):
            reasoning.append(
                {
                    "reasoning_id": "final:main",
                    "scope": "main",
                    "start_seq": None,
                    "end_seq": None,
                    "text": str(response["reasoning_content"]),
                }
            )

        return {
            "card": {
                "status": str(response.get("status") or "failed"),
                "latency_ms": int(
                    (perf_counter() - self._started_at) * 1000
                ),
                "iterations": self._iterations or len(reasoning),
                "model": self._model,
                "provider": self._provider,
                "reasoning": reasoning,
                "tools": [
                    self._public_tool(item)
                    for item in self._ordered(self._tools.values())
                ],
                "subagents": [
                    self._public_subagent(item)
                    for item in self._ordered(self._subagents.values())
                ],
                "memory": self._memory,
                "artifact_refs": list(
                    response.get("artifact_refs") or []
                ),
                "pending_action": response.get("pending_action"),
                "error": response.get("error"),
            },
            "schema_version": 1,
        }

    def _observe_reasoning(
        self,
        data: dict[str, Any],
        sequence: int,
        timestamp: str,
    ) -> None:
        reasoning_id = str(
            data.get("reasoning_id")
            or data.get("delegation_id")
            or "main"
        )
        item = self._reasoning.setdefault(
            reasoning_id,
            {
                "reasoning_id": reasoning_id,
                "scope": str(data.get("scope") or "main"),
                "delegation_id": data.get("delegation_id"),
                "workflow": data.get("workflow"),
                "start_seq": sequence,
                "end_seq": sequence,
                "started_at": timestamp,
                "finished_at": timestamp,
                "text": "",
            },
        )
        item["end_seq"] = sequence
        item["finished_at"] = timestamp
        item["text"] = self._append_text(
            str(item["text"]),
            str(data.get("delta") or ""),
        )

    def _observe_action_started(
        self,
        data: dict[str, Any],
        sequence: int,
        timestamp: str,
    ) -> None:
        action_id = str(data.get("action_id") or "")
        if not action_id:
            return

        action_type = str(data.get("action_type") or "tool")
        if action_type == "subagent":
            item = self._subagent_for(
                data,
                sequence,
                timestamp,
            )
        else:
            item = self._tool_for(data, sequence, timestamp)

        if data.get("requires_confirmation"):
            item["status"] = "awaiting_approval"

    def _observe_action_result(
        self,
        data: dict[str, Any],
        sequence: int,
        timestamp: str,
    ) -> None:
        action_type = str(data.get("action_type") or "tool")
        if action_type == "subagent":
            item = self._subagent_for(data, sequence, timestamp)
        else:
            item = self._tool_for(data, sequence, timestamp)

        item.update(
            {
                "status": str(data.get("status") or "failed"),
                "end_seq": sequence,
                "finished_at": timestamp,
                "duration_ms": self._duration_ms(item),
                "summary": data.get("summary"),
                "artifact_refs": list(data.get("artifact_refs") or []),
                "retryable": bool(data.get("retryable")),
                "error_code": data.get("error_code"),
                "error_message": data.get("error_message"),
            }
        )

    def _observe_memory(
        self,
        event: str,
        data: dict[str, Any],
        sequence: int,
        timestamp: str,
    ) -> None:
        event_status = event.removeprefix("memory_retrieval_")
        status = "running" if event_status == "started" else event_status
        current = self._memory or {
            "status": "running",
            "facts_count": 0,
            "episodes_count": 0,
            "start_seq": sequence,
            "started_at": timestamp,
        }
        current.update(
            {
                "status": status,
                "facts_count": int(data.get("facts_count") or 0),
                "episodes_count": int(data.get("episodes_count") or 0),
                "end_seq": sequence,
                "finished_at": timestamp,
            }
        )
        if event_status == "started":
            current["end_seq"] = None
            current["finished_at"] = None
        self._memory = current

    def _observe_subagent_event(
        self,
        event: str,
        data: dict[str, Any],
        sequence: int,
        timestamp: str,
    ) -> None:
        item = self._subagent_for(data, sequence, timestamp)
        result_data = to_jsonable(data.get("data"))
        item.update(
            {
                "phase": data.get("phase"),
                "phase_label": data.get("phase_label"),
                "progress_percent": data.get("progress_percent", data.get("progress")),
                "iteration": data.get("iteration"),
                "last_seq": sequence,
                "last_updated_at": timestamp,
                "task_id": data.get("task_id"),
                "warnings": list(data.get("warnings") or []),
                "error": data.get("error"),
            }
        )
        status = data.get("status")
        if event == "subagent_started":
            item["status"] = "running"
        elif event == "subagent_completed":
            item["status"] = str(status or "success")
        elif event == "subagent_failed":
            item["status"] = str(status or "error")
        elif status:
            item["status"] = str(status)

        if event in {"subagent_completed", "subagent_failed"} and isinstance(
            result_data,
            dict,
        ):
            item["result"] = result_data

    def _observe_timeline_step(
        self,
        data: dict[str, Any],
        sequence: int,
        timestamp: str,
    ) -> None:
        item = self._subagent_for(data, sequence, timestamp)
        step_id = str(data.get("step_id") or "")
        if not step_id:
            return

        steps = item.setdefault("_timeline", {})
        step = steps.setdefault(
            step_id,
            {
                "step_id": step_id,
                "step_key": data.get("step_key"),
                "label": data.get("label"),
                "iteration": data.get("iteration"),
                "state": "running",
                "start_seq": sequence,
                "end_seq": None,
                "started_at": timestamp,
                "finished_at": None,
                "duration_ms": None,
                "error": None,
                "_started_at": perf_counter(),
            },
        )
        state = str(data.get("state") or "running")
        step["state"] = state
        if state in {"completed", "failed"}:
            step["end_seq"] = sequence
            step["finished_at"] = timestamp
            step["duration_ms"] = self._duration_ms(step)
            step["error"] = data.get("error")

    def _tool_for(
        self,
        data: dict[str, Any],
        sequence: int,
        timestamp: str,
    ) -> dict[str, Any]:
        action_id = str(data.get("action_id") or "")
        return self._tools.setdefault(
            action_id,
            {
                "action_id": action_id,
                "name": data.get("name"),
                "status": "running",
                "start_seq": sequence,
                "end_seq": None,
                "started_at": timestamp,
                "finished_at": None,
                "duration_ms": None,
                "input": to_jsonable(data.get("input") or {}),
                "summary": None,
                "artifact_refs": [],
                "retryable": False,
                "error_code": None,
                "error_message": None,
                "_started_at": perf_counter(),
            },
        )

    def _subagent_for(
        self,
        data: dict[str, Any],
        sequence: int,
        timestamp: str,
    ) -> dict[str, Any]:
        action_id = data.get("delegation_id") or data.get("action_id")
        key = str(action_id or data.get("workflow") or "subagent")
        item = self._subagents.setdefault(
            key,
            {
                "delegation_id": action_id,
                "workflow": data.get("workflow"),
                "name": data.get("name") or data.get("subagent"),
                "status": "running",
                "start_seq": sequence,
                "end_seq": None,
                "started_at": timestamp,
                "finished_at": None,
                "duration_ms": None,
                "input": to_jsonable(data.get("input") or {}),
                "timeline": [],
                "_timeline": {},
                "_started_at": perf_counter(),
            },
        )
        if action_id and item.get("delegation_id") is None:
            item["delegation_id"] = action_id
        if data.get("workflow"):
            item["workflow"] = data["workflow"]
        if data.get("name") or data.get("subagent"):
            item["name"] = data.get("name") or data.get("subagent")
        return item

    @staticmethod
    def _append_text(current: str, delta: str) -> str:
        max_length = 8000
        return (current + delta)[:max_length]

    @staticmethod
    def _ordered(items: Any) -> list[dict[str, Any]]:
        return sorted(
            items,
            key=lambda item: int(item.get("start_seq") or 0),
        )

    @staticmethod
    def _duration_ms(item: dict[str, Any]) -> int:
        started_at = item.get("_started_at")
        if not isinstance(started_at, float):
            return 0
        return int((perf_counter() - started_at) * 1000)

    @staticmethod
    def _public_reasoning(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in item.items()
            if key not in {"delegation_id", "workflow"}
            or value is not None
        }

    @staticmethod
    def _public_tool(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in item.items()
            if not key.startswith("_")
        }

    def _public_subagent(self, item: dict[str, Any]) -> dict[str, Any]:
        result = {
            key: value
            for key, value in item.items()
            if not key.startswith("_") and key != "timeline"
        }
        result["timeline"] = [
            self._public_tool(step)
            for step in self._ordered(item.get("_timeline", {}).values())
        ]
        return result
