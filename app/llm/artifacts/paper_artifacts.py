from __future__ import annotations

import uuid
from typing import Any

from app.llm.artifacts.schemas import ArtifactRef, PaperSearchArtifactPayload
from app.llm.artifacts.store import LocalArtifactStore


async def save_paper_search_artifact(
    *,
    source: str,
    run_id: str | None,
    step_key: str,
    query: str | None,
    search_query: str | None,
    ok: bool,
    papers: list[dict[str, Any]],
    total_results: int = 0,
    metadata: dict[str, Any] | None = None,
) -> ArtifactRef:
    """
    保存论文检索结果。

    arXiv / DBLP / Google Scholar 都调用这个函数。
    """

    real_run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"

    payload = PaperSearchArtifactPayload(
        ok=ok,
        source=source,
        run_id=real_run_id,
        step_key=step_key,
        query=query,
        search_query=search_query,
        total_results=total_results,
        returned_count=len(papers),
        papers=papers,
        metadata=metadata or {},
    )

    store = LocalArtifactStore()

    artifact = await store.write_json(
        run_id=real_run_id,
        step_key=step_key,
        source=source,
        kind="paper_search_result_json",
        payload=payload.model_dump(),
        count=len(papers),
        name_hint=search_query or query or source,
        metadata={
            "query": query,
            "search_query": search_query,
            "source": source,
            **(metadata or {}),
        },
    )

    return artifact


def build_tool_return_with_artifact(
    *,
    base_result: dict[str, Any],
    artifact: ArtifactRef | None,
    return_papers: bool,
) -> dict[str, Any]:
    result = dict(base_result)

    artifact_refs: list[str] = []
    metadata: dict[str, Any] = {
        "source": result.get("source"),
        "run_id": result.get("run_id"),
        "step_key": result.get("step_key"),
    }

    if artifact is not None:
        result["artifact_uri"] = artifact.artifact_uri
        result["json_path"] = artifact.path
        result["artifact_kind"] = artifact.kind

        artifact_refs.append(artifact.artifact_uri)
        metadata.update(
            {
                "run_id": artifact.run_id,
                "step_key": artifact.step_key,
                "artifact_kind": artifact.kind,
                "json_path": artifact.path,
            }
        )

    if not return_papers:
        result.pop("papers", None)
        result["papers_omitted"] = True

    ok = bool(result.get("ok", False))
    error = None

    if not ok:
        error = str(
            result.get("error")
            or result.get("message")
            or "arXiv search failed"
        )

    return {
        "ok": ok,
        "result": result if ok else None,
        "error": error,
        "artifact_refs": artifact_refs,
        "metadata": {
            key: value
            for key, value in metadata.items()
            if value is not None
        },
    }