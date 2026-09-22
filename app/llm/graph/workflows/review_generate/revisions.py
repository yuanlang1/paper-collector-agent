from app.llm.graph.workflows.review_generate.contracts import content_hash, save
from app.llm.graph.workflows.review_generate.nodes.finalize import failed


async def snapshot(store, state):
    values = []
    for key in (
        "claims_artifact_ref",
        "evidence_ledger_artifact_ref",
        "framework_artifact_ref",
        "review_draft_artifact_ref",
    ):
        value = await store.read_json_uri(state[key]) if state.get(key) else None
        if value and key == "evidence_ledger_artifact_ref":
            value = [
                {"claim_id": item["claim_id"], "chunk_snippets": item["chunk_snippets"]}
                for item in value["claims"]
            ]
        if value and key == "claims_artifact_ref":
            value = [
                {"claim_id": item["claim_id"], "claim_hash": item["claim_hash"]}
                for item in value["claims"]
            ]
        values.append(value)
    return content_hash(values)


async def schedule_revision(store, state, items, *, claim_ids, section_ids):
    allowed = {
        "claim": set(claim_ids),
        "retrieval": set(claim_ids),
        "section": set(section_ids),
        "framework": {"review"},
        "synthesis": {"review"},
    }
    if any(item["target_id"] not in allowed[item["target_type"]] for item in items):
        raise ValueError("revision contains an unknown target")
    current = await snapshot(store, state)
    ref = await save(store, state, "revision_plan", {"items": items, "before": current})
    update = {"revision_plan_artifact_ref": ref, "revision_items": items}
    if not items or current == state.get("revision_before"):
        return failed(
            "revision produced no actionable progress", **update, error_code="REVIEW_NO_PROGRESS"
        )
    if state["reflection_round"] >= state["max_reflection_rounds"] - 1:
        return failed(
            "review revision budget exhausted", **update, error_code="REVIEW_QUALITY_FAILED"
        )
    kinds = {item["target_type"] for item in items}
    stage = next(
        stage
        for kind, stage in (
            ("claim", "generating_claims"),
            ("retrieval", "retrieving_evidence"),
            ("framework", "generating_framework"),
            ("section", "rendering_sections"),
            ("synthesis", "assembling_review"),
        )
        if kind in kinds
    )
    return {
        **update,
        "revision_before": current,
        "reflection_round": state["reflection_round"] + 1,
        "stage": stage,
        "status": "running",
    }
