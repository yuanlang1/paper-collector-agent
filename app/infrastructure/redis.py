import logging

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import settings

logger = logging.getLogger(__name__)


class RedisManager:
    def __init__(
        self,
        *,
        url: str,
        connect_timeout: float = 1,
        socket_timeout: float = 1,
        max_connections: int = 20,
    ) -> None:
        self._url = url.strip()
        self._connect_timeout = connect_timeout
        self._socket_timeout = socket_timeout
        self._max_connections = max_connections
        self._client: Redis | None = None

    @property
    def client(self) -> Redis | None:
        return self._client

    async def start(self) -> None:
        if not self._url or self._client is not None:
            return
        self._client = Redis.from_url(
            self._url,
            decode_responses=True,
            socket_connect_timeout=self._connect_timeout,
            socket_timeout=self._socket_timeout,
            max_connections=self._max_connections,
        )
        if not await self.ping():
            logger.warning("Redis startup check failed; later operations can reconnect")

    async def ping(self) -> bool:
        if self._client is None:
            return False
        try:
            return bool(await self._client.ping())
        except RedisError:
            return False

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()


redis_manager = RedisManager(
    url=settings.REDIS_URL,
    connect_timeout=settings.REDIS_CONNECT_TIMEOUT_SECONDS,
    socket_timeout=settings.REDIS_SOCKET_TIMEOUT_SECONDS,
    max_connections=settings.REDIS_MAX_CONNECTIONS,
)
