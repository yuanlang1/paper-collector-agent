from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator


_STREAM_FINISHED = object()


class DetachedStreamRun:
    """A single SSE subscriber attached to work that outlives the request."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._events: asyncio.Queue[object] = asyncio.Queue()
        self._subscriber_connected = True

    def publish(self, event: str) -> None:
        if self._subscriber_connected:
            self._events.put_nowait(event)

    def finish(self) -> None:
        if self._subscriber_connected:
            self._events.put_nowait(_STREAM_FINISHED)

    async def subscribe(self) -> AsyncIterator[str]:
        try:
            while True:
                event = await self._events.get()

                if event is _STREAM_FINISHED:
                    return
                yield str(event)
        finally:
            self._subscriber_connected = False
