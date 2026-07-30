from typing import Any

from langgraph.config import get_stream_writer


def emit_custom_event(
    payload: dict[str, Any],
) -> None:
    """
    向 LangGraph custom stream 写入 UI 临时事件。

    这些事件不进入 State，也不会被 Checkpointer 持久化。
    """
    try:
        writer = get_stream_writer()

    except RuntimeError:
        # 单元测试或脱离 graph 调用工具时，
        # 不存在 stream writer。
        return

    writer(payload)


def emit_tool_event(
    event: str,
    *,
    tool_name: str,
    display_name: str | None = None,
    todo_id: str | None = None,
    message: str | None = None,
    progress: int | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    emit_custom_event(
        {
            "event": event,
            "tool_name": tool_name,
            "display_name": display_name,
            "todo_id": todo_id,
            "message": message,
            "progress": progress,
            "data": data or {},
        }
    )


def emit_tool_started(
    tool_name: str,
    *,
    display_name: str | None = None,
    todo_id: str | None = None,
    message: str | None = None,
) -> None:
    emit_tool_event(
        "tool_started",
        tool_name=tool_name,
        display_name=display_name,
        todo_id=todo_id,
        message=message,
        progress=0,
    )


def emit_tool_progress(
    tool_name: str,
    progress: int,
    *,
    display_name: str | None = None,
    todo_id: str | None = None,
    message: str | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    emit_tool_event(
        "tool_progress",
        tool_name=tool_name,
        display_name=display_name,
        todo_id=todo_id,
        message=message,
        progress=max(0, min(progress, 100)),
        data=data,
    )


def emit_tool_completed(
    tool_name: str,
    *,
    display_name: str | None = None,
    todo_id: str | None = None,
    message: str | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    emit_tool_event(
        "tool_completed",
        tool_name=tool_name,
        display_name=display_name,
        todo_id=todo_id,
        message=message,
        progress=100,
        data=data,
    )


def emit_tool_failed(
    tool_name: str,
    *,
    display_name: str | None = None,
    todo_id: str | None = None,
    message: str | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    emit_tool_event(
        "tool_failed",
        tool_name=tool_name,
        display_name=display_name,
        todo_id=todo_id,
        message=message,
        data=data,
    )


def emit_todo_progress(
    progress: int,
    *,
    todo_id: str | None = None,
    message: str | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    """
    用于没有工具调用的 TODO 内部进度，
    例如合并、去重、排序。
    """
    emit_custom_event(
        {
            "event": "todo_progress",
            "todo_id": todo_id,
            "message": message,
            "progress": max(0, min(progress, 100)),
            "data": data or {},
        }
    )