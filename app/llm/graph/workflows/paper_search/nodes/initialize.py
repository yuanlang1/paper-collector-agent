from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from app.llm.subagents.paper_search.contracts import PaperSearchDelegation


async def initialize_paper_search_node(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    call = state.get("active_tool_call")
    request_payload = call.get("args") if isinstance(call, Mapping) else None
    if not isinstance(request_payload, Mapping):
        return {
            "stage": "failed",
            "status": "failed",
            "error": "论文检索子图缺少有效委派请求。",
        }

    try:
        request = PaperSearchDelegation.model_validate(request_payload)
    except ValidationError as exc:
        return {
            "stage": "failed",
            "status": "failed",
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
