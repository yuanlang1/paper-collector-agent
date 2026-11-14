import asyncio
import logging
from pathlib import Path

from celery import group, states

from app.config import settings
from app.core.identifiers import normalize_user_id, validate_run_id
from app.tasks.celery_app import celery_app
from app.tasks.celery_utils import run_async_task

logger = logging.getLogger(__name__)
COUNT_KEYS = (
    "source_abstract",
    "extracted_abstract",
    "missing_abstract",
    "front_pages_read",
    "ai_abstract",
    "source_keyword",
    "extracted_keyword",
    "missing_keyword",
)


@celery_app.task(
    name="paper_search.enrich_abstract", queue="paper_search", track_started=True
)
def enrich_abstract(
    *, user_id: str, child_run_id: str, paper_index: int, paper: dict
) -> dict:
    async def execute(db):
        from app.services import paper_abstract_service

        numeric_user_id = normalize_user_id(user_id)[1]
        validate_run_id(child_run_id)
        path = paper.get("pdf_download", {}).get("local_pdf_path")
        if path:
            root = (Path(settings.ARTIFACT_BASE_DIR) / child_run_id).resolve()
            Path(path).resolve().relative_to(root)
        async with asyncio.timeout(settings.PAPER_ABSTRACT_TASK_TIMEOUT_SECONDS):
            result = await paper_abstract_service.enrich_paper_abstract(
                db,
                user_id=numeric_user_id,
                paper=paper,
            )
        return {"paper_index": paper_index, **result}

    return run_async_task(f"{child_run_id}:{paper_index}", "paper_abstract", execute)


def _failure(paper, error):
    from app.services.paper_abstract_service import (
        _keywords_text,
        _normalize_keywords,
        _text,
    )

    info = paper["paper_info"]
    abstract = _text(info.get("paper_abstract"))
    keywords = _keywords_text(_normalize_keywords(info.get("keywords")))
    counts = dict.fromkeys(COUNT_KEYS, 0)
    counts["source_abstract" if abstract else "missing_abstract"] = 1
    counts["source_keyword" if keywords else "missing_keyword"] = 1
    resolution = {
        "source": info.get("abstract_source") or "source_metadata"
        if abstract
        else "missing",
        "extraction_status": "success" if abstract else "failed",
        "pdf_pages_read": 0,
        "ai_summary_source": "failed",
        "error": str(error),
    }
    return {
        "paper_info_patch": {
            "paper_abstract": abstract,
            "abstract_source": resolution["source"],
            "keywords": keywords,
            "ai_abstract": None,
        },
        "abstract_resolution": resolution,
        "keyword_resolution": {
            "source": "source_metadata" if keywords else "missing",
            "extraction_status": "fallback" if keywords else "failed",
            "pdf_pages_read": 0,
            "error": str(error),
        },
        "counts": counts,
        "warning": f"{info.get('title')} 的摘要补全警告：{error}",
    }


async def enrich_abstracts(papers, *, user_id, child_run_id):
    counts = dict.fromkeys(COUNT_KEYS, 0)
    warnings = []
    if not papers:
        return {"papers": papers, "counts": counts, "warnings": warnings}
    job = group(
        enrich_abstract.s(
            user_id=str(user_id),
            child_run_id=child_run_id,
            paper_index=index,
            paper={
                "paper_info": {
                    key: paper["paper_info"].get(key)
                    for key in (
                        "title",
                        "paper_abstract",
                        "abstract_source",
                        "keywords",
                    )
                },
                "pdf_download": {
                    "local_pdf_path": (paper.get("pdf_download") or {}).get(
                        "local_pdf_path"
                    )
                },
            },
        )
        for index, paper in enumerate(papers)
    )
    result = job.freeze()
    pending = dict(enumerate(result.results))
    try:
        async with asyncio.timeout(settings.PAPER_ABSTRACT_GROUP_TIMEOUT_SECONDS):
            # Finish publication before handling cancellation so revocation cannot race late publication.
            publishing = asyncio.create_task(asyncio.to_thread(job.apply_async))
            try:
                await asyncio.shield(publishing)
            except asyncio.CancelledError:
                await publishing
                raise
            while pending:

                def collect():
                    return [
                        (index, task.state, task.result)
                        for index, task in list(pending.items())
                        if task.ready()
                    ]

                for index, status, payload in await asyncio.to_thread(collect):
                    paper = papers[index]
                    try:
                        if status != states.SUCCESS:
                            raise ValueError(str(payload) or status)
                        if payload["paper_index"] != index:
                            raise ValueError("摘要任务索引不匹配")
                        patch = payload["paper_info_patch"]
                        values = {key: payload["counts"][key] for key in COUNT_KEYS}
                        if any(
                            type(value) is not int or value < 0
                            for value in values.values()
                        ):
                            raise ValueError("摘要任务统计无效")
                        for key in (
                            "paper_abstract",
                            "abstract_source",
                            "ai_abstract",
                            "keywords",
                        ):
                            if patch[key] is not None and not isinstance(
                                patch[key], str
                            ):
                                raise ValueError("摘要任务字段无效")
                        for key in ("abstract_resolution", "keyword_resolution"):
                            if not isinstance(payload[key], dict):
                                raise ValueError("摘要任务来源记录无效")
                        if payload["warning"] is not None and not isinstance(
                            payload["warning"], str
                        ):
                            raise ValueError("摘要任务警告无效")
                    except (KeyError, TypeError, ValueError) as exc:
                        payload = _failure(paper, exc)
                    paper["paper_info"].update(
                        {
                            key: payload["paper_info_patch"][key]
                            for key in (
                                "paper_abstract",
                                "abstract_source",
                                "ai_abstract",
                                "keywords",
                            )
                        }
                    )
                    for key in ("abstract_resolution", "keyword_resolution"):
                        paper[key] = payload[key]
                    for key in COUNT_KEYS:
                        counts[key] += payload["counts"][key]
                    if payload["warning"]:
                        warnings.append(payload["warning"])
                    await asyncio.to_thread(pending.pop(index).forget)
                if pending:
                    await asyncio.sleep(0.5)
    except BaseException:
        try:
            await asyncio.to_thread(result.revoke, terminate=False)
        except Exception:
            logger.exception("Failed to revoke abstract group %s", result.id)
        raise
    return {"papers": papers, "counts": counts, "warnings": warnings}
