import json
from enum import Enum
from typing import Any


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump(mode="json"))

    if isinstance(value, dict):
        return {
            key: to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            to_jsonable(item) for item in value
        ]

    return value


def encode_sse(
    event: str,
    data: dict[str, Any],
) -> str:
    payload = json.dumps(
        to_jsonable(data),
        ensure_ascii=False,
        default=str,
    )

    return (
        f"event: {event}\n"
        f"data: {payload}\n\n"
    )


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return ""

    parts: list[str] = []

    for item in content:
        if isinstance(item, str):
            parts.append(item)

        elif (isinstance(item, dict) and item.get("type") == "text"):
            parts.append(item.get("text", ""))

    return "".join(parts)


def extract_interrupt_payload(
    chunk: Any,
) -> Any | None:
    if not isinstance(chunk, dict):
        return None

    interrupts = chunk.get("__interrupt__")

    if interrupts is None:
        for value in chunk.values():
            if (isinstance(value, dict) and "__interrupt__" in value):
                interrupts = value["__interrupt__"]
                break

    if not interrupts:
        return None

    interrupt = (
        interrupts[0]
        if isinstance(interrupts, (list, tuple))
        else interrupts
    )

    return getattr(interrupt, "value", interrupt,)


def is_user_visible_message(
    metadata: dict[str, Any],
) -> bool:
    node_name = metadata.get("langgraph_node")
    tags = set(metadata.get("tags") or [])

    return node_name == "final" or "user_visible" in tags