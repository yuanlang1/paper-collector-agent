from __future__ import annotations

import grpc
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.protos.agent.v1 import agent_pb2, agent_pb2_grpc
from app.services import internal_settings_service as settings_service


def _limits_view(values: dict[str, dict[str, int]]) -> agent_pb2.SourceLimitsView:
    limits = values["limits"]
    page_sizes = values["page_sizes"]
    maximums = values["maximum_limits"]
    return agent_pb2.SourceLimitsView(
        arxiv=limits["arxiv"],
        dblp=limits["dblp"],
        google_scholar=limits["google_scholar"],
        arxiv_page_size=page_sizes["arxiv"],
        dblp_page_size=page_sizes["dblp"],
        google_scholar_page_size=page_sizes["google_scholar"],
        arxiv_maximum=maximums["arxiv"],
        dblp_maximum=maximums["dblp"],
        google_scholar_maximum=maximums["google_scholar"],
    )


class SettingsGrpcService(agent_pb2_grpc.SettingsServiceServicer):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def GetSourceLimits(self, request, context):
        async with self.session_factory() as db:
            values = await settings_service.get_limits(db)
        return _limits_view(values)

    async def UpdateSourceLimits(self, request, context):
        try:
            limits = {
                "arxiv": request.arxiv,
                "dblp": request.dblp,
                "google_scholar": request.google_scholar,
            }
            async with self.session_factory() as db:
                values = await settings_service.update_limits(db, limits)
            return _limits_view(values)
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))


def add_services(
    server: grpc.aio.Server,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    agent_pb2_grpc.add_SettingsServiceServicer_to_server(
        SettingsGrpcService(session_factory), server,
    )
