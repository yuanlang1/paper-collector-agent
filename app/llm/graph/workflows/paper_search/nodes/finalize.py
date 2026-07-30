from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.llm.subagents.paper_search.contracts import (
    build_paper_search_handoff,
)


async def finalize_paper_search_node(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    handoff = build_paper_search_handoff(state)
    return {
        "paper_search_handoff": handoff.model_dump(mode="json"),
    }
