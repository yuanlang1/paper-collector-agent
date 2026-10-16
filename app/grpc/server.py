from __future__ import annotations

import grpc
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import settings
from app.grpc.services import add_services


class InternalGrpcServer:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._server = grpc.aio.server(
            options=[
                ("grpc.max_receive_message_length", settings.AGENT_GRPC_MAX_MESSAGE_LENGTH),
                ("grpc.max_send_message_length", settings.AGENT_GRPC_MAX_MESSAGE_LENGTH),
            ]
        )
        add_services(self._server, session_factory)
        self._address = f"{settings.AGENT_GRPC_HOST}:{settings.AGENT_GRPC_PORT}"

    async def start(self) -> None:
        if self._server.add_insecure_port(self._address) == 0:
            raise RuntimeError(f"Unable to bind internal gRPC server: {self._address}")
        await self._server.start()

    async def stop(self) -> None:
        await self._server.stop(grace=5)
