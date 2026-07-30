from typing import Any

from app.llm.streaming.utils import (
    to_jsonable,
)


def _string_value(
    value: Any,
) -> str | None:
    if value is None:
        return None

    if hasattr(value, "value"):
        return str(value.value)

    return str(value)


def _interrupt_value(
    state: dict[str, Any],
) -> Any | None:
    interrupts = state.get(
        "__interrupt__"
    )

    if not interrupts:
        return None

    interrupt = (
        interrupts[0]
        if isinstance(
            interrupts,
            (list, tuple),
        )
        else interrupts
    )

    return getattr(
        interrupt,
        "value",
        interrupt,
    )


def _status(
    state: dict[str, Any],
    interrupt: Any | None,
) -> str:
    if interrupt is not None:
        return "confirmation_required"

    run_status = _string_value(
        state.get("run_status")
    )

    if run_status in {
        "running",
        "completed",
        "failed",
        "blocked",
    }:
        return run_status

    if state.get("error"):
        return "failed"

    return "completed"


def _reply(
    state: dict[str, Any],
    interrupt: Any | None,
) -> str:
    if interrupt is not None:
        normalized = to_jsonable(
            interrupt
        )

        if isinstance(
            normalized,
            dict,
        ):
            message = normalized.get(
                "message"
            )

            if message:
                return str(message)

        return "需要确认后继续执行。"

    reply = state.get("reply")

    if reply:
        return str(reply)

    error = state.get("error")

    if error:
        return f"任务执行失败：{error}"

    return ""


def _pending_action(
    state: dict[str, Any],
) -> dict[str, Any] | None:
    pending = to_jsonable(
        state.get("pending_action")
    )

    if not isinstance(pending, dict):
        return None

    action_type = pending.get("action")

    if action_type == "tool":
        name = pending.get("tool_name")
        action_input = (
            pending.get("tool_arguments")
            or {}
        )
    elif action_type == "subagent":
        name = pending.get("subagent_name")
        action_input = (
            pending.get("subagent_input")
            or {}
        )
    else:
        return None

    action_id = pending.get(
        "action_id"
    )

    if not action_id or not name:
        return None

    return {
        "action_id": str(action_id),
        "action_type": action_type,
        "name": str(name),
        "input": (
            action_input
            if isinstance(
                action_input,
                dict,
            )
            else {}
        ),
        "requires_confirmation": bool(
            pending.get(
                "requires_confirmation"
            )
        ),
        "message": pending.get(
            "confirmation_message"
        ),
    }


def _last_action_result(
    state: dict[str, Any],
) -> dict[str, Any] | None:
    result = to_jsonable(
        state.get("last_action_result")
    )

    if not isinstance(result, dict):
        return None

    action_id = result.get(
        "action_id"
    )
    action_type = result.get(
        "action_type"
    )
    name = result.get("name")
    status = result.get("status")

    if not all(
        [
            action_id,
            action_type,
            name,
            status,
        ]
    ):
        return None

    data = result.get("data")
    artifact_refs = result.get(
        "artifact_refs"
    )

    return {
        "action_id": str(action_id),
        "action_type": str(
            action_type
        ),
        "name": str(name),
        "status": str(status),
        "summary": str(
            result.get("summary")
            or ""
        ),
        "data": (
            data
            if isinstance(data, dict)
            else {}
        ),
        "artifact_refs": [
            str(item)
            for item in (
                artifact_refs
                if isinstance(
                    artifact_refs,
                    list,
                )
                else []
            )
        ],
        "retryable": bool(
            result.get("retryable")
        ),
        "error_code": result.get(
            "error_code"
        ),
        "error_message": result.get(
            "error_message"
        ),
    }


def _artifact_refs(
    state: dict[str, Any],
) -> list[str]:
    value = to_jsonable(
        state.get("artifact_refs")
        or []
    )

    if not isinstance(value, list):
        return []

    return [
        str(item)
        for item in value
    ]


def _error(
    state: dict[str, Any],
) -> str | None:
    state_error = state.get("error")

    if state_error:
        return str(state_error)

    if (
        _string_value(
            state.get("run_status")
        )
        != "failed"
    ):
        return None

    result = _last_action_result(
        state
    )

    if result:
        error_message = result.get(
            "error_message"
        )

        if error_message:
            return str(error_message)

    return "任务执行失败"


def _build_payload(
    *,
    state: dict[str, Any],
    conversation_id: str,
    fallback_run_id: str | None,
) -> dict[str, Any]:
    run_id = (
        state.get("run_id")
        or fallback_run_id
    )

    if not run_id:
        raise ValueError(
            "MainAgentState 缺少 run_id"
        )

    interrupt = _interrupt_value(
        state
    )

    return {
        "conversation_id": (
            conversation_id
        ),
        "run_id": str(run_id),
        "status": _status(
            state,
            interrupt,
        ),
        "reply": _reply(
            state,
            interrupt,
        ),
        "pending_action": (
            _pending_action(state)
        ),
        "last_action_result": (
            _last_action_result(state)
        ),
        "artifact_refs": (
            _artifact_refs(state)
        ),
        "interrupt": (
            to_jsonable(interrupt)
            if interrupt is not None
            else None
        ),
        "error": _error(state),
    }


def build_chat_response(
    result: dict[str, Any],
    conversation_id: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    return _build_payload(
        state = result,
        conversation_id = conversation_id,
        fallback_run_id = run_id,
    )


def build_done_payload(
    state: dict[str, Any],
    conversation_id: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    return _build_payload(
        state=state,
        conversation_id=(
            conversation_id
        ),
        fallback_run_id=run_id,
    )
