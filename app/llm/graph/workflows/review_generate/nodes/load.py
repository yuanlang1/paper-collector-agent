from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, Field
from app.infrastructure.grpc.paper_service_grpc_client import (
    PaperServiceGrpcClient,
    paper_service_grpc_client,
)
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.nodes.finalize import failed

RagStatus = Literal["ready", "pending", "skipped", "failed", "indexing"]
RAG_STATUSES = {"ready", "pending", "skipped", "failed", "indexing"}

class CorpusPaper(BaseModel):
    paper_id: int = Field(..., gt = 0)
    title: str = ""
    abstract: str = ""
    authors: list[str] = Field(default_factory = list)
    year: int | None = Field(default = None, ge = 1800, le = 2100)
    doi: str | None = None
    oss_name: str | None = None
    pdf_url: str | None = None

    rag_status: RagStatus = "pending"
    chunk_count: int = Field(default = 0, ge = 0)

class TaskReviewCorpus(BaseModel):
    task_id: int = Field(..., gt = 0)
    papers: list[CorpusPaper] = Field(default_factory = list)

class LoadTaskCorpusNode:
    def __init__(
        self,
        client: PaperServiceGrpcClient | None = None,
        artifact_store: LocalArtifactStore | None = None,
    ) -> None:
        self.client = client or paper_service_grpc_client
        self.store = artifact_store or LocalArtifactStore()

    async def __call__(
        self, 
        state: dict
    ) -> dict:
        if state.get("stage") != "loading_corpus":
            return failed(
                "load_task_corpus called in invalid stage"
            )

        run_id = state.get("run_id")
        task_id = state.get("task_id")

        if not isinstance(run_id, str) or not run_id.strip():
            return failed("missing valid run_id")

        if (
            not isinstance(task_id, int)
            or isinstance(task_id, bool)
            or task_id <= 0
        ):
            return failed("missing valid task_id")

        response = await self.client.get_task_review_papers(task_id)

        if not response["ok"]:
            return failed(response["error"] or "failed to load task review papers")

        result = response["result"]
        if not isinstance(result, dict):
            return failed("task review papers response is invalid")


        if result.get("task_id") != task_id:
            return failed("task review papers returned a mismatched task_id")

        raw_papers = result.get("papers")
        if not isinstance(raw_papers, list):
            return failed("task review papers payload is invalid")

        if not raw_papers:
            return failed("task has no associated papers")

        papers: list[dict[str, Any]] = []
        seen_paper_ids: set[int] = set()
        duplicate_count = 0

        for raw_paper in raw_papers:
            if not isinstance(raw_paper, dict):
                return failed("task review paper item is invalid")

            paper_id = raw_paper.get("paper_id")
            if (
                not isinstance(paper_id, int)
                or isinstance(paper_id, bool)
                or paper_id <= 0
            ):
                return failed("task review paper_id is invalid")

            if paper_id in seen_paper_ids:
                duplicate_count += 1
                continue

            rag_status = str(raw_paper.get("rag_status") or "").strip().lower()

            if rag_status not in RAG_STATUSES:
                return failed(
                    f"paper_id = {paper_id} has invalid "
                    f"rag_status = {rag_status!r}"
                )

            chunk_count = raw_paper.get("chunk_count")
            if (
                not isinstance(chunk_count, int)
                or isinstance(chunk_count, bool)
                or chunk_count < 0
            ):
                return failed(f"paper_id = {paper_id} has invalid chunk_count")

            if rag_status == "ready" and chunk_count == 0:
                return failed(f"paper_id = {paper_id} is ready but has no chunks")

            seen_paper_ids.add(paper_id)

            papers.append(
                {
                    "paper_id": paper_id,
                    "title": str(raw_paper.get("title") or "").strip(),
                    "abstract": str(
                        raw_paper.get("paper_abstract") or ""
                    ).strip(),
                    "authors": [
                        str(author).strip()
                        for author in raw_paper.get("authors", [])
                        if str(author).strip()
                    ],
                    "published_date": raw_paper.get("published_date"),
                    "doi": raw_paper.get("doi"),
                    "oss_name": str(
                        raw_paper.get("oss_name") or ""
                    ).strip() or None,
                    "pdf_url": str(
                        raw_paper.get("pdf_url") or ""
                    ).strip() or None,
                    "rag_status": rag_status,
                    "chunk_count": chunk_count,
                }
            )

        rag_status_counts = Counter(
            paper["rag_status"]
            for paper in papers
        )
        task_status = str(result.get("task_status") or "").strip().lower()

        warnings = list(state.get("warnings", []))

        if duplicate_count:
            warnings.append(f"Removed {duplicate_count} duplicate task-paper relation(s).")

        if (
            task_status != "rag_completed"
            or rag_status_counts["pending"]
            or rag_status_counts["indexing"]
        ):
            return {
                "stage": "blocked",
                "status": "blocked",
                "error": None,
                "warnings": [
                    *warnings,
                    "RAG indexing is not complete; review generation is blocked.",
                ],
            }

        review_papers = [
            paper
            for paper in papers
            if paper["rag_status"] == "ready"
        ]

        if not review_papers:
            return {
                "stage": "blocked",
                "status": "blocked",
                "error": None,
                "warnings": [
                    *warnings,
                    "RAG completed but no paper has usable full-text evidence.",
                ],
            }

        if len(review_papers) != len(papers):
            warnings.append(
                "Excluded failed or skipped papers from the review corpus."
            )

        paper_ids_snapshot = [
            str(paper["paper_id"])
            for paper in review_papers
        ]

        artifact = await self.store.write_json(
            run_id = run_id,
            step_key = "load_task_corpus",
            source = "task_review",
            kind = "task_review_corpus_json",
            count = len(review_papers),
            payload = {
                "task_id": task_id,
                "task_status": task_status,
                "paper_ids_snapshot": paper_ids_snapshot,
                "papers": review_papers,
                "summary": {
                    "total_associated_papers": len(papers),
                    "review_eligible_papers": len(review_papers),
                    "rag_status_counts": dict(rag_status_counts),
                    "duplicate_relations_removed": duplicate_count,
                },
            },
        )

        return {
            "paper_ids_snapshot": paper_ids_snapshot,
            "corpus_artifact_ref": artifact.artifact_uri,

            "warnings": warnings,
            "error": None,
            "stage": "extracting_studies",
            "status": "running",
        }
