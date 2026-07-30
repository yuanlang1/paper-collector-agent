from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from app.llm.subagents.paper_search.contracts import (
    PaperSearchDelegation,
)


async def initialize_paper_search_node(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    request_payload = state.get("paper_search_request")

    if request_payload is None:
        prompt = state.get("original_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return {
                "stage": "blocked",
                "status": "blocked",
                "error": "论文检索子图缺少有效请求。",
            }
        request_payload = {
            "prompt": prompt,
            "constraints": state.get("paper_search_constraints") or {},
        }

    try:
        request = PaperSearchDelegation.model_validate(request_payload)
    except ValidationError as exc:
        return {
            "stage": "blocked",
            "status": "blocked",
            "error": f"论文检索请求无效：{exc}",
        }

    return {
        "original_prompt": request.prompt,
        "paper_search_constraints": request.constraints.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "warnings": [],
        "progress": {},
        "downloaded_pdf_paths": [],
        "pdf_cleanup_error": None,
        "degraded": False,
        "stage": "intent_understanding",
        "status": "running",
        "error": None,
    }
