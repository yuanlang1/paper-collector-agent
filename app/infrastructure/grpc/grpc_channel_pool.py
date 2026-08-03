import asyncio
from typing import Any

import grpc

from app.config import settings
from app.infrastructure.nacos_registry import nacos_registry


def _get_instance_attr(instance: Any, name: str) -> Any:
    if isinstance(instance, dict):
        return instance.get(name)

    return getattr(instance, name, None)


class PaperServiceGrpcChannelPool:
    def __init__(self) -> None:
        self._channels: dict[str, grpc.aio.Channel] = {}
        self._lock = asyncio.Lock()

    async def get_channel(self) -> grpc.aio.Channel:
        instance = await nacos_registry.choose_instance(
            settings.PAPER_SERVICE_GRPC_NAME
        )

        ip = _get_instance_attr(instance, "ip")
        port = _get_instance_attr(instance, "port")

        if not ip or not port:
            raise RuntimeError(
                "Nacos 返回的 paper-service-grpc 实例缺少 ip 或 port"
            )

        target = f"{ip}:{port}"

        async with self._lock:
            channel = self._channels.get(target)

            if channel is None:
                channel = grpc.aio.insecure_channel(
                    target,
                    options=[
                        (
                            "grpc.max_receive_message_length",
                            settings.PAPER_SERVICE_GRPC_MAX_RECEIVE_MESSAGE_LENGTH,
                        ),
                        ("grpc.keepalive_time_ms", 30_000),
                        ("grpc.keepalive_timeout_ms", 5_000),
                    ],
                )
                self._channels[target] = channel

            return channel

    async def close(self) -> None:
        async with self._lock:
            channels = list(self._channels.values())
            self._channels.clear()

        for channel in channels:
            await channel.close()


paper_service_grpc_channel_pool = PaperServiceGrpcChannelPool()