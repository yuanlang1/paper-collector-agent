from __future__ import annotations

import uuid
from typing import Any

from app.llm.artifacts.schemas import ArtifactRef
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
    """保存论文检索结果。"""

    real_run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"

    # ponytail: 沿用现有 JSON 结构，直接用字典保存。
    payload = {
        "ok": ok,
        "source": source,
        "run_id": real_run_id,
        "step_key": step_key,
        "query": query,
        "search_query": search_query,
        "total_results": total_results,
        "returned_count": len(papers),
        "papers": papers,
        "metadata": metadata or {},
    }

    store = LocalArtifactStore()

    return await store.write_json(
        run_id=real_run_id,
        step_key=step_key,
        source=source,
        payload=payload,
    )
