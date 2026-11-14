import asyncio
import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from celery import group, states

from app.config import settings
from app.llm.tools.search_tools.crossref.search_crossref import crossref_search_service
from app.tasks.celery_app import celery_app, get_worker_loop

logger = logging.getLogger(__name__)


def optional_text(value: Any) -> str | None:
    if value is None:
        return None

    value = str(value).strip()
    return value or None


def normalize_doi(value: Any) -> str | None:
    doi = optional_text(value)
    if doi is None:
        return None

    return re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        doi,
        flags=re.IGNORECASE,
    ).strip() or None


def _normalize_title(value: Any) -> str:
    return re.sub(
        r"[^\w\s]",
        "",
        str(value or "").casefold(),
    ).replace(" ", "")


def _first_author(value: Any) -> str | None:
    if isinstance(value, list):
        return next(
            (
                author
                for item in value
                if (author := optional_text(item)) is not None
            ),
            None,
        )

    authors = optional_text(value)
    if authors is None:
        return None

    return re.split(r"\s*(?:,|;|\band\b)\s*", authors, 1)[0] or None


def _author_matches(expected: Any, actual: Any) -> bool:
    expected_author = _first_author(expected)
    actual_author = _first_author(actual)

    if expected_author is None or actual_author is None:
        return True

    expected_tokens = set(re.findall(r"\w+", expected_author.casefold()))
    actual_tokens = set(re.findall(r"\w+", actual_author.casefold()))
    return bool(expected_tokens & actual_tokens)


def _matching_candidate(
    paper_info: Mapping[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    title = _normalize_title(paper_info.get("title"))

    if not title:
        return None

    for candidate in candidates:
        if (
            _normalize_title(candidate.get("title")) == title
            and _author_matches(
                paper_info.get("authors"),
                candidate.get("authors"),
            )
        ):
            return candidate

    return None


def _response_papers(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    if response.get("ok") is not True:
        error = response.get("error")
        message = (
            error.get("message")
            if isinstance(error, Mapping)
            else error
        )
        raise RuntimeError(str(message or "Crossref search failed."))

    papers = response.get("papers")
    return [paper for paper in papers if isinstance(paper, dict)] if isinstance(papers, list) else []


async def search_crossref_paper(
    paper_info: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    doi = normalize_doi(paper_info.get("doi"))
    if doi:
        response = await crossref_search_service({"doi": doi}, None)
        papers = _response_papers(response)
        if papers:
            return (
                papers[0],
                "doi",
            )

    title = optional_text(paper_info.get("title"))
    if title is None:
        return None, "title"

    params: dict[str, Any] = {"title": title, "limit": 5}
    first_author = _first_author(paper_info.get("authors"))
    if first_author:
        params["author"] = first_author

    response = await crossref_search_service(params, None)
    papers = _response_papers(response)
    return (
        _matching_candidate(
            paper_info,
            papers,
        ),
        "title",
    )


@celery_app.task(name="paper_search.query_crossref", queue="paper_search")
def query_crossref(
    *, paper_index: int, doi: str | None, title: str | None, authors: str | None,
) -> dict:
    async def execute():
        async with asyncio.timeout(settings.PAPER_CROSSREF_TASK_TIMEOUT_SECONDS):
            candidate, method = await search_crossref_paper(
                {"doi": doi, "title": title, "authors": authors}
            )
        return {"paper_index": paper_index, "candidate": candidate, "match_method": method}

    return get_worker_loop().run_until_complete(execute())


async def query_papers(papers: list[dict[str, Any]]) -> dict[int, dict]:
    if not papers:
        return {}

    timeout = settings.PAPER_CROSSREF_GROUP_TIMEOUT_SECONDS
    expires = datetime.now(UTC) + timedelta(seconds=timeout)
    job = group(
        query_crossref.s(
            paper_index=index,
            doi=info.get("doi"),
            title=info.get("title"),
            authors=info.get("authors"),
        )
        for index, info in enumerate(papers)
    )
    group_result = job.freeze()
    pending = dict(enumerate(group_result.results))
    results = {}

    def collect_ready():
        return [(index, task.state, task.result) for index, task in pending.items() if task.ready()]

    try:
        async with asyncio.timeout(timeout):
            # ponytail: expire queued work if a publisher thread outlives the group deadline.
            published = await asyncio.to_thread(job.apply_async, expires=expires)
            pending = dict(enumerate(published.results))
            while pending:
                finished = []
                for index, status, payload in await asyncio.to_thread(collect_ready):
                    if status != states.SUCCESS:
                        payload = {"error": str(payload) or status}
                    elif (
                        not isinstance(payload, dict)
                        or type(payload.get("paper_index")) is not int
                        or payload["paper_index"] != index
                        or payload.get("match_method") not in ("doi", "title")
                        or "candidate" not in payload
                        or (payload["candidate"] is not None and not isinstance(payload["candidate"], dict))
                    ):
                        payload = {"error": "Crossref 任务结果无效"}
                    results[index] = payload
                    finished.append(pending.pop(index))
                for task in finished:
                    try:
                        await asyncio.to_thread(task.forget)
                    except Exception:
                        logger.warning("Could not forget Crossref result %s", task.id, exc_info=True)
                if pending:
                    await asyncio.sleep(0.5)
    except (Exception, asyncio.CancelledError) as exc:
        try:
            await asyncio.wait_for(
                asyncio.to_thread(group_result.revoke, terminate=False), timeout=3,
            )
        except Exception:
            logger.warning("Could not revoke Crossref group %s", group_result.id, exc_info=True)
        if isinstance(exc, asyncio.CancelledError):
            raise
        error = "Crossref 查询批次超时" if isinstance(exc, TimeoutError) else str(exc) or type(exc).__name__
        results.update({index: {"error": error} for index in pending})

    return results
