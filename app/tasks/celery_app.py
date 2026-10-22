import os
from importlib.metadata import version

from celery import Celery

celery_app = Celery(
    "paper_collector_memory",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://127.0.0.1:6380/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://127.0.0.1:6380/1"),
)

celery_app.conf.update(
    task_default_queue="memory",
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
    broker_connection_retry_on_startup=True,
    worker_prefetch_multiplier=1,
    timezone="Asia/Shanghai",
)


@celery_app.task(name="memory.healthcheck")
def healthcheck() -> dict[str, str]:
    return {
        "status": "ok",
        "langmem": version("langmem"),
        "qdrant_client": version("qdrant-client"),
    }
