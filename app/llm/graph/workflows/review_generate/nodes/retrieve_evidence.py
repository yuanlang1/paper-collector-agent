from app.llm.artifacts.store import LocalArtifactStore
from app.rag.retrieval.paper_content_retrieval import PaperContentHybridRetrievalModule
from app.llm.graph.workflows.review_generate.contracts import (
    retrieval_plan_hash,
    review_batch,
    save,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import failed


MAX_EVIDENCE_QUERIES = 2
RETRIEVAL_TOP_K = 6
MAX_EVIDENCE_SNIPPETS = 6
MAX_EVIDENCE_SNIPPETS_PER_PAPER = 2
EVIDENCE_SELECTION_VERSION = "rrf-bm25-top6-v2"


def _snippet(document):
    metadata = document.metadata
    return {
        "paper_id": str(metadata["paper_id"]),
        "chunk_id": str(metadata["chunk_id"]),
        "text": document.page_content,
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


def select_evidence_snippets(query_results):
    candidates = {}
    ranked_keys = []
    for documents in query_results:
        keys = []
        for rank, document in enumerate(documents, start=1):
            snippet = _snippet(document)
            key = snippet["paper_id"], snippet["chunk_id"]
            if key not in candidates:
                candidates[key] = {
                    "snippet": snippet,
                    "score": 0.0,
                    "first_seen": len(candidates),
                }
            candidates[key]["score"] += 1 / (60 + rank)
            keys.append(key)
        ranked_keys.append(keys)

    selected = []
    paper_counts = {}

    def add(key):
        if key in selected or len(selected) == MAX_EVIDENCE_SNIPPETS:
            return False
        paper_id = candidates[key]["snippet"]["paper_id"]
        if paper_counts.get(paper_id, 0) == MAX_EVIDENCE_SNIPPETS_PER_PAPER:
            return False
        selected.append(key)
        paper_counts[paper_id] = paper_counts.get(paper_id, 0) + 1
        return True

    for keys in ranked_keys:
        for key in keys:
            if add(key):
                break

    ordered = sorted(
        candidates,
        key=lambda key: (-candidates[key]["score"], candidates[key]["first_seen"]),
    )
    for key in ordered:
        if candidates[key]["snippet"]["paper_id"] not in paper_counts:
            add(key)
    for key in ordered:
        add(key)
    return [candidates[key]["snippet"] for key in selected]


class RetrieveEvidenceNode:
    def __init__(self, *, artifact_store=None, content_retrieval=None):
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.content_retrieval = content_retrieval

    async def prepare(self, state):
        claims = (await self.artifact_store.read_json_uri(state["claims_artifact_ref"]))["claims"]
        previous = {}
        if state.get("evidence_ledger_artifact_ref"):
            previous = {
                item["claim_id"]: item
                for item in (await self.artifact_store.read_json_uri(
                    state["evidence_ledger_artifact_ref"]
                ))["claims"]
            }
        jobs, claim_ids, cached = [], [], {}
        user_id = state["user_id"]
        for claim in claims:
            if claim["withdrawn_reason"]:
                continue
            claim_id = claim["claim_id"]
            claim_ids.append(claim_id)
            queries = claim["retrieval_queries"][:MAX_EVIDENCE_QUERIES]
            plan_hash = retrieval_plan_hash(
                claim,
                queries=queries,
                settings={
                    "top_k": RETRIEVAL_TOP_K,
                    "user_id": user_id,
                    "selection_version": EVIDENCE_SELECTION_VERSION,
                },
            )
            old = previous.get(claim_id)
            if (
                old
                and old["claim_hash"] == claim["claim_hash"]
                and old.get("retrieval_plan_hash") == plan_hash
            ):
                cached[claim_id] = old
                continue
            jobs.append({
                "item_id": claim_id,
                "claim_hash": claim["claim_hash"],
                "retrieval_plan_hash": plan_hash,
                "queries": queries,
                "paper_ids": claim["candidate_paper_ids"],
                "user_id": user_id,
            })
        return review_batch(jobs, claim_ids=claim_ids, cached=cached)

    async def run_item(self, item):
        if self.content_retrieval is None:
            self.content_retrieval = PaperContentHybridRetrievalModule()
        query_results = []
        for query in item["queries"]:
            query_results.append(await self.content_retrieval.search(
                query=query,
                user_id=item["user_id"],
                top_k=RETRIEVAL_TOP_K,
                paper_ids=item["paper_ids"],
            ))
        return {
            "claim_id": item["item_id"],
            "claim_hash": item["claim_hash"],
            "retrieval_plan_hash": item["retrieval_plan_hash"],
            "queries": item["queries"],
            "chunk_snippets": select_evidence_snippets(query_results),
        }

    async def collect(self, state, results, failure):
        if failure:
            return self.failure(state, failure)
        context = state["batch_context"]
        entries = {**context["cached"], **results}
        ref = await save(
            self.artifact_store, state, "evidence_ledger",
            {"claims": [entries[claim_id] for claim_id in context["claim_ids"]]},
        )
        return {
            "evidence_ledger_artifact_ref": ref,
            "stage": "verifying_claims",
            "status": "running",
        }

    @staticmethod
    def failure(state, error):
        return failed(
            f"evidence retrieval failed: {error['error']}",
            error_code="RETRIEVAL_FAILED", retryable=True,
        )
