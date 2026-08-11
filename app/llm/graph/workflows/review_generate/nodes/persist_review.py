from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.infrastructure.grpc.review_service_grpc_client import (
    ReviewServiceGrpcClient,
    review_service_grpc_client,
)
from app.llm.artifacts.store import LocalArtifactStore


class PersistReviewNode:
    def __init__(
        self,
        *,
        artifact_store: LocalArtifactStore | None = None,
        client: ReviewServiceGrpcClient | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.client = client or review_service_grpc_client

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        if state.get("stage") != "persisting_review":
            return {
                "stage": "failed",
                "status": "failed",
                "error": "persist_review called in invalid stage",
            }

        final_review_artifact_ref = state.get("final_review_artifact_ref")
        if not isinstance(final_review_artifact_ref, str):
            return {
                "stage": "failed",
                "status": "failed",
                "error": "missing final review artifact",
            }

        try:
            final_review = await self.artifact_store.read_json_uri(
                final_review_artifact_ref
            )
        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"failed to load final review artifact: {exc}",
            }

        result = await self.client.add_review(
            {
                "task_id": state["task_id"],
                "topic": state["topic"],
                "title": final_review["title"],
                "version_number": state.get("review_version_number", 1),
                "language": state["language"],
                "review_type": state["review_type"],
                "citation_style": state["citation_style"],
                "scope": final_review["scope"],
                "abstract_content": final_review["abstract"],
                "body_markdown": final_review["body_markdown"],
                "conclusion": final_review["conclusion"],
                "markdown": final_review["markdown"],
                "sections": final_review["sections"],
                "paper_ids_snapshot": state["paper_ids_snapshot"],
                "citation_paper_ids": final_review["citation_paper_ids"],
                "citation_labels": final_review["citation_labels"],
                "references": final_review["references"],
                "framework_hash": state.get("framework_hash"),
            }
        )

        if not result["ok"]:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"review persistence failed: {result['error']}",
            }

        review_id = result["result"]["review_id"]
        return {
            "review_id": review_id,
            "review_version_number": state.get("review_version_number", 1),
            "handoff": {
                "action": "review_saved",
                "review_id": review_id,
                "task_id": state["task_id"],
            },
            "stage": "completed",
            "status": "completed",
            "error": None,
        }
