from typing import Any, Counter, Literal, Mapping

from pydantic import BaseModel, Field
from app.infrastructure.grpc import task_paper_relation_grpc_client
from app.infrastructure.grpc.task_paper_relation_grpc_client import TaskPaperRelationGrpcClient
from app.llm.artifacts.store import LocalArtifactStore

RagStatus = Literal["ready", "pending", "skipped", "failed", "indexing"]

class CorpusPaper(BaseModel):
    paper_id: int = Field(..., gt = 0)
    title: str = ""
    abstract: str = ""
    authors: list[str] = Field(default_factory = list)
    year: int | None = Field(default = None, ge = 1800, le = 2100)
    doi: str | None = None

    rag_status: RagStatus = "pending"
    chunk_count: int = Field(default = 0, ge = 0)

class TaskReviewCorpus(BaseModel):
    task_id: int = Field(..., gt = 0)
    papers: list[CorpusPaper] = Field(default_factory = list)

def _failed(error: str) -> dict:
    return {
        "stage": "failed",
        "status": "failed",
        "error": error,
    }



class LoadTaskCorpusNode:
    def __init__(
        self,
        client: TaskPaperRelationGrpcClient | None = None,
        artifact_store: LocalArtifactStore | None = None,
    ) -> None:
        self.client = client or task_paper_relation_grpc_client
        self.store = artifact_store or LocalArtifactStore()

    async def __call__(
        self, 
        state: dict
    ) -> dict:
        if state.get("stage") != "loading_corpus":
            return _failed(
                "load_task_corpus called in invalid stage"
            )

        run_id = state.get("run_id")
        task_id = state.get("task_id")

        if not isinstance(run_id, str) or not run_id.strip():
            return _failed("missing valid run_id")

        if (
            not isinstance(task_id, int)
            or isinstance(task_id, bool)
            or task_id <= 0
        ):
            return _failed("missing valid task_id")

        response = await self.client.get_task_review_papers(task_id)

        if not response["ok"]:
            return _failed(response["error"] or "failed to load task review papers")

        result = response["result"]
        if not isinstance(result, dict):
            return _failed("task review papers response is invalid")


        if result.get("task_id") != task_id:
            return _failed("task review papers returned a mismatched task_id")

        raw_papers = result.get("papers")
        if not isinstance(raw_papers, list):
            return _failed("task review papers payload is invalid")

        if not raw_papers:
            return _failed("task has no associated papers")

        papers: list[dict[str, Any]] = []
        seen_paper_ids: set[int] = set()
        duplicate_count = 0

        for raw_paper in raw_papers:
            if not isinstance(raw_paper, dict):
                return _failed("task review paper item is invalid")

            paper_id = raw_paper.get("paper_id")
            if (
                not isinstance(paper_id, int)
                or isinstance(paper_id, bool)
                or paper_id <= 0
            ):
                return _failed("task review paper_id is invalid")

            if paper_id in seen_paper_ids:
                duplicate_count += 1
                continue

            rag_status = str(raw_paper.get("rag_status") or "").strip().lower()

            if rag_status not in RagStatus:
                return _failed(
                    f"paper_id = {paper_id} has invalid "
                    f"rag_status = {rag_status!r}"
                )

            chunk_count = raw_paper.get("chunk_count")
            if (
                not isinstance(chunk_count, int)
                or isinstance(chunk_count, bool)
                or chunk_count < 0
            ):
                return _failed(f"paper_id = {paper_id} has invalid chunk_count")

            if rag_status == "ready" and chunk_count == 0:
                return _failed(f"paper_id = {paper_id} is ready but has no chunks")

            seen_paper_ids.add(paper_id)

            papers.append(
                {
                    "paper_id": paper_id,
                    "title": str(raw_paper.get("title") or "").strip(),
                    "authors": [
                        str(author).strip()
                        for author in raw_paper.get("authors", [])
                        if str(author).strip()
                    ],
                    "published_date": raw_paper.get("published_date"),
                    "doi": raw_paper.get("doi"),
                    "rag_status": rag_status,
                    "chunk_count": chunk_count,
                }
            )

        paper_ids_snapshot = [
            str(paper["paper_id"])
            for paper in papers
        ]
        rag_status_counts = Counter(
            paper["rag_status"]
            for paper in papers
        )

        warnings = list(state.get("warnings", []))

        if duplicate_count:
            warnings.append(f"Removed {duplicate_count} duplicate task-paper relation(s).")

        if rag_status_counts["ready"] == 0:
            warnings.append("No task paper has ready full-text RAG coverage.")

        artifact = await self.artifact_store.write_json(
            run_id = run_id,
            step_key = "load_task_corpus",
            source = "task_review",
            kind = "task_review_corpus_json",
            count = len(papers),
            payload = {
                "task_id": task_id,
                "task_status": str(
                    result.get("task_status") or ""
                ),
                "paper_ids_snapshot": paper_ids_snapshot,
                "papers": papers,
                "summary": {
                    "total_papers": len(papers),
                    "rag_status_counts": dict(rag_status_counts),
                    "duplicate_relations_removed": duplicate_count,
                },
            },
        )

        return {
            "paper_ids_snapshot": paper_ids_snapshot,
            "corpus_artifact_ref": artifact.artifact_uri,
            "corpus_coverage_artifact_ref": None,

            "warnings": warnings,
            "error": None,
            "stage": "assessing_rag_coverage",
            "status": "running",
        }




        
        