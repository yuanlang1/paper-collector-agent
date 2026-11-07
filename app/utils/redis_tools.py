import json
from typing import Any
from urllib.parse import quote

from redis.asyncio import Redis

from app.config import settings


def build_key(namespace: str, *parts: str | int, prefix: str | None = None) -> str:
    prefix = settings.REDIS_KEY_PREFIX if prefix is None else prefix
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("Redis key prefix must not be blank")
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("Redis key namespace must not be blank")
    if any(type(part) not in (str, int) for part in parts):
        raise TypeError("Redis key parts must be strings or integers")
    return ":".join(quote(str(part), safe="") for part in (prefix, namespace, *parts))


async def get_json(client: Redis, key: str) -> Any | None:
    payload = await client.get(key)
    if payload is None:
        return None
    value = json.loads(payload)
    if value is None:
        raise ValueError("JSON null is not a supported cache value")
    return value


async def set_json(client: Redis, key: str, value: Any, *, ttl_seconds: int) -> None:
    if type(ttl_seconds) is not int or ttl_seconds < 1:
        raise ValueError("ttl_seconds must be a positive integer")
    if value is None:
        raise ValueError("JSON null is not a supported cache value")
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False)
    await client.set(key, payload, ex=ttl_seconds)


async def delete_keys(client: Redis, *keys: str) -> int:
    if not keys:
        return 0
    return await client.delete(*keys)
