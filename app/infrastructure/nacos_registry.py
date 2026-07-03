import logging
import random
import socket
from typing import Any

from v2.nacos import (
    ClientConfigBuilder,
    DeregisterInstanceParam,
    GRPCConfig,
    ListInstanceParam,
    NacosNamingService,
    RegisterInstanceParam,
)

from app.config import settings

logger = logging.getLogger(__name__)


def get_local_ip() -> str:
    if settings.SERVICE_IP:
        return settings.SERVICE_IP

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]

class NacosRegistry:
    def __init__(self) -> None:
        self.client = None
        self.service_ip = get_local_ip()
        self.service_port = settings.SERVICE_PORT
    
    def _build_client_config(self):
        builder = (
            ClientConfigBuilder()
            .server_address(settings.NACOS_SERVER_ADDR)
            .namespace_id(settings.NACOS_NAMESPACE_ID)
            .log_level(settings.NACOS_LOG_LEVEL)
            .grpc_config(GRPCConfig(grpc_timeout=settings.NACOS_GRPC_TIMEOUT_MS))
        )

        if settings.NACOS_USERNAME:
            builder.username(settings.NACOS_USERNAME)

        if settings.NACOS_PASSWORD:
            builder.password(settings.NACOS_PASSWORD)

        return builder.build()

    async def start(self) -> None:
        if not settings.NACOS_ENABLED:
            logger.info("Nacos registration disabled")
            return

        self.client = await NacosNamingService.create_naming_service(
            self._build_client_config()
        )

        await self.client.register_instance(
            request=RegisterInstanceParam(
                service_name=settings.SERVICE_NAME,
                group_name=settings.NACOS_GROUP_NAME,
                ip=self.service_ip,
                port=self.service_port,
                weight=1.0,
                cluster_name=settings.NACOS_CLUSTER_NAME,
                metadata={
                    "framework": "fastapi",
                    "version": settings.APP_VERSION,
                    "health": "/health",
                },
                enabled=True,
                healthy=True,
                ephemeral=True,
            )
        )

        logger.info(
            "Registered service to Nacos: service=%s, group=%s, namespace=%s, address=%s:%s",
            settings.SERVICE_NAME,
            settings.NACOS_GROUP_NAME,
            settings.NACOS_NAMESPACE_ID,
            self.service_ip,
            self.service_port,
        )

    async def stop(self) -> None:
        if not self.client:
            return

        await self.client.deregister_instance(
            request=DeregisterInstanceParam(
                service_name=settings.SERVICE_NAME,
                group_name=settings.NACOS_GROUP_NAME,
                ip=self.service_ip,
                port=self.service_port,
                cluster_name=settings.NACOS_CLUSTER_NAME,
                ephemeral=True,
            )
        )

        await self.client.shutdown()
        logger.info("Deregistered service from Nacos: %s", settings.SERVICE_NAME)

    async def list_instances(self, service_name: str) -> list[Any]:
        if not self.client:
            raise RuntimeError("Nacos naming client is not initialized")

        return await self.client.list_instances(
            ListInstanceParam(
                service_name=service_name,
                group_name=settings.NACOS_GROUP_NAME,
                clusters=[settings.NACOS_CLUSTER_NAME],
                healthy_only=True,
            )
        )

    async def choose_instance(self, service_name: str) -> Any:
        instances = await self.list_instances(service_name)

        candidates = [
            instance
            for instance in instances
            if getattr(instance, "healthy", True)
            and getattr(instance, "enabled", True)
            and getattr(instance, "weight", 1) > 0
        ]

        if not candidates:
            raise RuntimeError(f"No healthy instance found for service: {service_name}")

        return random.choice(candidates)


nacos_registry = NacosRegistry()

