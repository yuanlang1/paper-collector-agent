import logging
from uuid import UUID

from celery import states
from fastapi import APIRouter
from kombu.exceptions import KombuError
from redis.exceptions import RedisError

from app.common.exceptions import ServiceUnavailableException
from app.common.response import ServiceResponse
from app.schemas.rag import IndexBatchRequest
from app.tasks.celery_app import celery_app
from app.tasks.modules.rag.index_batch import index_batch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/rag", tags=["internal-rag"])


@router.post("/batches", status_code=202)
def submit_batch(body: IndexBatchRequest):
    try:
        job = index_batch.delay(paper_ids=body.paper_ids)
    except (KombuError, RedisError, OSError):
        logger.exception("Unable to submit RAG batch")
        raise ServiceUnavailableException("索引任务提交失败，请稍后重试。") from None

    return ServiceResponse.build_success_response({"job_id": job.id})


@router.get("/batches/{job_id}")
def get_batch(job_id: UUID):
    job_id = str(job_id)
    try:
        result = celery_app.AsyncResult(job_id)
        status = result.state
        payload = (
            result.result
            if status == states.SUCCESS or status in states.EXCEPTION_STATES
            else None
        )
    except (KombuError, RedisError, OSError):
        raise ServiceUnavailableException("索引任务结果暂不可用。") from None

    return ServiceResponse.build_success_response(
        {
            "job_id": job_id,
            "execution_status": status,
            "results": payload.get("results")
            if status == states.SUCCESS and isinstance(payload, dict)
            else None,
            "error": str(payload) if status in states.EXCEPTION_STATES else None,
        }
    )
