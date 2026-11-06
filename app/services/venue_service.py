import logging

from pydantic import ValidationError
from redis.exceptions import RedisError

from app.config import settings
from app.infrastructure.redis import redis_manager
from app.infrastructure.paper_service.venue_client import (
    VenueServiceClient,
    venue_service_client,
)
from app.schemas.venue import VenueRecord
from app.utils.redis_key import VENUE_KEY_TEMPLATE
from app.utils.redis_tools import get_json, set_json

logger = logging.getLogger(__name__)


class VenueService:
    def __init__(
        self, *, client: VenueServiceClient | None = None
    ):
        self.client = client or venue_service_client

    async def get_by_name(self, standard_name: str) -> tuple[VenueRecord | None, str]:
        redis = redis_manager.client
        name = standard_name.strip()
        if not name or len(name) > 500:
            raise ValueError("standard_name must contain 1 to 500 characters")
        key = VENUE_KEY_TEMPLATE.format(standard_name=name)
        if redis is not None:
            try:
                cached = await get_json(redis, key)
                if cached is not None:
                    return VenueRecord.model_validate(cached), "redis_cache"
            except (RedisError, ValueError, TypeError):
                logger.warning("Venue cache read failed; querying paper service")

        response = await self.client.get_venue_by_name(name)
        if response.get("ok") is not True:
            logger.warning("Venue lookup failed: %s", response.get("error"))
            return None, "paper_service"
        payload = response["result"]["venue"]
        if payload is None:
            return None, "paper_service"
        try:
            record = VenueRecord.model_validate(payload)
        except ValidationError:
            logger.warning("Paper service returned invalid venue data")
            return None, "paper_service"
        if redis is not None:
            try:
                await set_json(
                    redis,
                    key,
                    record.model_dump(mode="json"),
                    ttl_seconds=settings.VENUE_CACHE_TTL_SECONDS,
                )
            except RedisError:
                logger.warning("Venue cache write failed")
        return record, "paper_service"
