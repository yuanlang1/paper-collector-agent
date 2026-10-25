from importlib.metadata import version

from app.tasks.celery_app import celery_app


@celery_app.task(name="memory.healthcheck")
def healthcheck() -> dict[str, str]:
    return {
        "status": "ok",
        "langmem": version("langmem"),
        "qdrant_client": version("qdrant-client"),
    }
