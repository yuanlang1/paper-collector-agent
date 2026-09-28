import asyncio

from app.llm.artifacts.store import LocalArtifactStore
from app.rag.retrieval.paper_content_retrieval import PaperContentHybridRetrievalModule
from app.llm.graph.workflows.review_generate.contracts import (
    retrieval_plan_hash,
    save,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import failed


class RetrieveEvidenceNode:
    def __init__(self, *, artifact_store=None, content_retrieval=None, max_concurrency=4):
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.content_retrieval = content_retrieval
        self.max_concurrency = max_concurrency

    async def __call__(self, state):
        try:
            claims = (await self.artifact_store.read_json_uri(state["claims_artifact_ref"]))[
                "claims"
            ]
            previous = {}
            if state.get("evidence_ledger_artifact_ref"):
                previous = {
                    item["claim_id"]: item
                    for item in (
                        await self.artifact_store.read_json_uri(
                            state["evidence_ledger_artifact_ref"]
                        )
                    )["claims"]
                }
            retrieval = self.content_retrieval or PaperContentHybridRetrievalModule()
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def retrieve(claim):
                old = previous.get(claim["claim_id"])
                plan_hash = retrieval_plan_hash(claim)
                if (
                    old
                    and old["claim_hash"] == claim["claim_hash"]
                    and old.get("retrieval_plan_hash") == plan_hash
                ):
                    return old
                snippets = {
                    (item["paper_id"], item["chunk_id"]): item
                    for item in (old or {}).get("chunk_snippets", [])
                }
                queries = claim["retrieval_queries"]
                async with semaphore:
                    for query in queries:
                        docs = await retrieval.search(
                            query=query,
                            top_k=10,
                            paper_ids=claim["candidate_paper_ids"],
                        )
                        for doc in docs:
                            metadata = doc.metadata
                            paper_id, chunk_id = (
                                str(metadata["paper_id"]),
                                str(metadata["chunk_id"]),
                            )
                            if paper_id not in state["paper_ids_snapshot"]:
                                raise ValueError("retrieval returned a paper outside task")
                            snippets[(paper_id, chunk_id)] = {
                                "paper_id": paper_id,
                                "chunk_id": chunk_id,
                                "text": doc.page_content,
                                **{
                                    key: metadata.get(key)
                                    for key in (
                                        "page_start",
                                        "page_end",
                                        "page_numbers",
                                        "section_path",
                                        "chunk_index",
                                    )
                                },
                            }
                return {
                    "claim_id": claim["claim_id"],
                    "claim_hash": claim["claim_hash"],
                    "retrieval_plan_hash": plan_hash,
                    "queries": queries,
                    "chunk_snippets": list(snippets.values()),
                }

            results = await asyncio.gather(
                *(retrieve(claim) for claim in claims if not claim["withdrawn_reason"])
            )
            ref = await save(self.artifact_store, state, "evidence_ledger", {"claims": results})
            return {
                "evidence_ledger_artifact_ref": ref,
                "stage": "verifying_claims",
                "status": "running",
            }
        except Exception as exc:
            return failed(
                f"evidence retrieval failed: {exc}", error_code="RETRIEVAL_FAILED", retryable=True
            )
