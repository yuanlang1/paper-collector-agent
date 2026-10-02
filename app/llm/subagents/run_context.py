from __future__ import annotations

import hashlib
from uuid import uuid4


def create_child_run_id(parent_run_id: str) -> str:
    """Create an artifact-safe ID scoped to one subgraph invocation."""
    parent_digest = hashlib.sha256(parent_run_id.encode("utf-8")).hexdigest()[:16]
    return f"child_{parent_digest}_{uuid4().hex}"
