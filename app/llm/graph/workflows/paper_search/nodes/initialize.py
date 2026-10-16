from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from app.llm.subagents.paper_search import PaperSearchParams


async def initialize_paper_search_node(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        request = PaperSearchParams.model_validate(state)
    except ValidationError as exc:
        return {
            "stage": "failed",
            "status": "failed",
            "error": f"论文检索请求无效：{exc}",
        }

    return {
        "original_prompt": request.prompt,
        "paper_service_task_id": request.task_id,
        "warnings": [],
        "progress": {},
        "downloaded_pdf_paths": [],
        "task_status_update_error": None,
        "remote_task_state": None,
        "stage": "preparing_search",
        "status": "running",
        "error": None,
    }
