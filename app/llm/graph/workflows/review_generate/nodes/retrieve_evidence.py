from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.nodes.finalize import failed
from app.rag.retrieval.paper_content_retrieval import PaperContentHybridRetrievalModule
from app.rag.retrieval.paper_retrieval import PaperHybridRetrievalModule


PAPER_TOP_K = 5
CHUNK_TOP_K = 10
MAX_CONCURRENCY = 4


class RetrieveEvidenceNode:
    def __init__(
        self,
        *,
        artifact_store: LocalArtifactStore | None = None,
        paper_retrieval: PaperHybridRetrievalModule | None = None,
        content_retrieval: (
            PaperContentHybridRetrievalModule | None
        ) = None,
        max_concurrency: int = MAX_CONCURRENCY,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.paper_retrieval = paper_retrieval
        self.content_retrieval = content_retrieval
        self.max_concurrency = max_concurrency

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            claims_payload, corpus_payload = await asyncio.gather(
                self.artifact_store.read_json_uri(
                    state["claims_artifact_ref"]
                ),
                self.artifact_store.read_json_uri(
                    state["corpus_artifact_ref"]
                ),
            )

            sections = claims_payload["claims"]["sections"]
            claims = [
                claim
                for section in sections
                for claim in section["claims"]
            ]
            retrieve_claim_ids = set(
                state.get("retrieve_claim_ids", [])
            )

            if state.get("evidence_ledger_artifact_ref") and not retrieve_claim_ids:
                return {
                    "stage": "rendering_sections",
                    "status": "running",
                }

            if self.paper_retrieval is None:
                self.paper_retrieval = PaperHybridRetrievalModule()
            if self.content_retrieval is None:
                self.content_retrieval = PaperContentHybridRetrievalModule()

            previous_evidence = {}
            if state.get("evidence_ledger_artifact_ref"):
                previous_payload = await self.artifact_store.read_json_uri(
                    state["evidence_ledger_artifact_ref"]
                )
                previous_evidence = {
                    item["claim_id"]: item
                    for item in previous_payload["claims"]
                }

            claims_to_retrieve = [
                claim
                for claim in claims
                if not retrieve_claim_ids
                or claim["claim_id"] in retrieve_claim_ids
            ]

            paper_ids_snapshot = state["paper_ids_snapshot"]
            titles_by_paper_id = {
                str(paper["paper_id"]): paper.get("title", "")
                for paper in corpus_payload["papers"]
            }

            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def retrieve_one(claim: dict[str, Any]) -> dict:
                async with semaphore:
                    return await self._retrieve_claim(
                        claim = claim,
                        paper_ids_snapshot = paper_ids_snapshot,
                        titles_by_paper_id = titles_by_paper_id,
                    )

            evidence_items = await asyncio.gather(
                *(retrieve_one(claim) for claim in claims_to_retrieve)
            )

            new_evidence = {
                item["claim_id"]: item
                for item in evidence_items
            }
            merged_evidence = [
                new_evidence.get(claim["claim_id"])
                or previous_evidence[claim["claim_id"]]
                for claim in claims
            ]

            affected_section_ids = [
                claim["section_id"]
                for claim in claims_to_retrieve
            ]

            artifact = await self.artifact_store.write_json(
                run_id = state["run_id"],
                step_key = (
                    f"retrieve_evidence_{state['reflection_round']}"
                    if retrieve_claim_ids
                    else "retrieve_evidence"
                ),
                source = "task_review",
                kind = "task_review_evidence_ledger_json",
                count = len(merged_evidence),
                payload = {
                    "task_id": state["task_id"],
                    "framework_hash": state["framework_hash"],
                    "paper_ids_snapshot": paper_ids_snapshot,
                    "claims": merged_evidence,
                },
            )

        except Exception as exc:
            return failed(f"evidence retrieval failed: {exc}")

        return {
            "evidence_ledger_artifact_ref": artifact.artifact_uri,
            "retrieve_claim_ids": [],
            "render_section_ids": list(
                dict.fromkeys(
                    [
                        *state.get("render_section_ids", []),
                        *affected_section_ids,
                    ]
                )
            ),
            "stage": "rendering_sections",
            "status": "running",
            "error": None,
        }

    async def _retrieve_claim(
        self,
        *,
        claim: dict[str, Any],
        paper_ids_snapshot: list[str],
        titles_by_paper_id: dict[str, str],
    ) -> dict[str, Any]:
        try:
            query = claim["rag_query"]

            paper_docs = await self.paper_retrieval.search(
                query = query,
                top_k = PAPER_TOP_K,
                paper_ids = paper_ids_snapshot,
            )

            candidate_paper_ids = list(
                dict.fromkeys(
                    str(document.metadata["paper_id"])
                    for document in paper_docs
                )
            )

            chunk_docs = await self.content_retrieval.search(
                query = query,
                top_k = CHUNK_TOP_K,
                paper_ids = (
                    candidate_paper_ids
                    or paper_ids_snapshot
                ),
            )

            support_papers = [
                {
                    "paper_id": str(document.metadata["paper_id"]),
                    "title": document.metadata.get("title", ""),
                    "abstract": document.metadata.get(
                        "abstract",
                        "",
                    ),
                    "score": document.metadata.get(
                        "retrieval_score",
                        0.0,
                    ),
                }
                for document in paper_docs
            ]

            chunk_snippets = []
            seen_chunks: set[tuple[str, str]] = set()

            for document in chunk_docs:
                metadata = document.metadata
                paper_id = str(metadata["paper_id"])
                chunk_id = str(metadata["chunk_id"])
                chunk_key = (paper_id, chunk_id)

                if chunk_key in seen_chunks:
                    continue

                seen_chunks.add(chunk_key)

                chunk_snippets.append(
                    {
                        "paper_id": paper_id,
                        "title": titles_by_paper_id.get(paper_id, ""),
                        "chunk_id": chunk_id,
                        "chunk_index": metadata.get("chunk_index"),
                        "section_path": metadata.get("section_path", ""),
                        "page_start": metadata.get("page_start"),
                        "page_end": metadata.get("page_end"),
                        "page_numbers": metadata.get("page_numbers", []),
                        "score": metadata.get("retrieval_score", 0.0),
                        "text": document.page_content,
                    }
                )

            return {
                "claim_id": claim["claim_id"],
                "section_id": claim["section_id"],
                "section_title": claim["section_title"],
                "text": claim["text"],
                "rag_query": query,
                "claim_type": claim["claim_type"],
                "evidence_requirement": (
                    claim["evidence_requirement"]
                ),
                "status": (
                    "retrieved"
                    if chunk_snippets
                    else "no_candidate"
                ),
                "support_papers": support_papers,
                "chunk_snippets": chunk_snippets,
                "retrieval_error": None,
            }

        except Exception as exc:
            return {
                "claim_id": claim["claim_id"],
                "section_id": claim["section_id"],
                "section_title": claim["section_title"],
                "text": claim["text"],
                "rag_query": claim["rag_query"],
                "claim_type": claim["claim_type"],
                "evidence_requirement": (
                    claim["evidence_requirement"]
                ),
                "status": "error",
                "support_papers": [],
                "chunk_snippets": [],
                "retrieval_error": str(exc),
            }

