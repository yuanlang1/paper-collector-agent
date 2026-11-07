import asyncio
import math
import random
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from pydantic import BaseModel

from app.llm.tools.search_tools.common import RETRYABLE_STATUS_CODES

MAX_TOOL_ATTEMPTS = 3
MAX_RETRY_DELAY_SECONDS = 30
RETRYABLE_TOOLS = {"search_web"}


def handler_tool_error(exc: ToolException | httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"工具请求失败，HTTP {exc.response.status_code}。"

    if isinstance(exc, httpx.HTTPError):
        return "工具网络请求失败，请稍后重试。"

    return str(exc) or "工具执行失败"


def _tool_output_preview(output: Any, limit: int = 200) -> str:
    if isinstance(output, ToolMessage):
        output = output.content

    text = output if isinstance(output, str) else str(output)

    return text if len(text) <= limit else text[:limit] + "…"


def _retry_delay(exc: ToolException | httpx.HTTPError, attempt: int) -> float | None:
    error = (
        (exc.__cause__ or exc.__context__) if isinstance(exc, ToolException) else exc
    )

    retry_after = None
    if isinstance(error, httpx.HTTPStatusError):
        if error.response.status_code not in RETRYABLE_STATUS_CODES:
            return None
        retry_after = error.response.headers.get("Retry-After")

    elif not isinstance(
        error, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
    ):
        return None

    delay = 2 ** (attempt - 1) + random.uniform(0, 0.2)
    if retry_after:
        try:
            requested_delay = float(retry_after)

        except ValueError:
            try:
                requested_delay = max(
                    0,
                    (
                        parsedate_to_datetime(retry_after) - datetime.now(UTC)
                    ).total_seconds(),
                )

            except (TypeError, ValueError, OverflowError):
                requested_delay = delay
        if math.isfinite(requested_delay) and requested_delay >= 0:
            delay = requested_delay

    return delay if delay <= MAX_RETRY_DELAY_SECONDS else None


async def awrap_tool_call(
    request: ToolCallRequest,
    execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
) -> ToolMessage | Command:
    call = request.tool_call
    metadata = request.runtime.config.get("metadata") or {}
    fields = {"name": call["name"], "tool_call_id": call["id"]}
    if parent_id := metadata.get("parent_tool_call_id"):
        fields["parent_tool_call_id"] = parent_id

    started = {"type": "tool_start", **fields}

    if call["name"] == "subagent":
        started["description"] = call["args"].get("description", "")

    writer = request.runtime.stream_writer
    writer(started)

    def write_progress(data: Any) -> None:
        if isinstance(data, BaseModel):
            data = data.model_dump(mode="json")
        writer({"parent_tool_call_id": call["id"], **data})

    request = replace(
        request,
        runtime=replace(
            request.runtime,
            config={
                **request.runtime.config,
                "metadata": {**metadata, "parent_tool_call_id": call["id"]},
            },
            stream_writer=write_progress,
        ),
    )

    max_attempts = MAX_TOOL_ATTEMPTS if call["name"] in RETRYABLE_TOOLS else 1
    for attempt in range(1, max_attempts + 1):
        try:
            result = await execute(request)

        except (ToolException, httpx.HTTPError) as exc:
            delay = _retry_delay(exc, attempt) if attempt < max_attempts else None

            if delay is not None:
                await asyncio.sleep(delay)
                continue

            result = ToolMessage(
                content=handler_tool_error(exc),
                name=call["name"],
                tool_call_id=call["id"],
                status="error",
            )

        except GraphBubbleUp:
            raise

        except Exception:
            writer(
                {
                    "type": "tool_failed",
                    **fields,
                    "status": "error",
                    "preview": "工具执行失败",
                }
            )
            raise

        status = result.status if isinstance(result, ToolMessage) else "success"
        writer(
            {
                "type": "tool_failed" if status == "error" else "tool_end",
                **fields,
                "status": status,
                "preview": _tool_output_preview(result),
            }
        )
        return result


def build_tool_node(tools):
    return ToolNode(
        tools=tools,
        handle_tool_errors=False,
        awrap_tool_call=awrap_tool_call,
    )
