from __future__ import annotations

from types import SimpleNamespace
import unittest

from langchain_core.documents import Document

from app.rag.evaluation.paper_content import (
    PaperContentCorpusReader,
    PaperContentRetrievalTarget,
)
from app.rag.evaluation.schemas import EvaluationCase, EvidenceRef


def _payload(chunk_index: int) -> dict:
    return {
        "paper_id": "120",
        "chunk_id": f"chunk-{chunk_index}",
        "chunk_index": chunk_index,
        "content": f"content {chunk_index}",
        "section_path": "Method",
        "page_start": 4,
        "page_end": 4,
        "page_numbers": [4],
        "source_spans": [
            {
                "page": 4,
                "type": "text",
                "source_index": chunk_index,
                "char_start": 0,
                "char_end": 9,
            }
        ],
    }


class _ScrollClient:
    async def scroll(self, *, offset, **_kwargs):
        if offset is None:
            return [SimpleNamespace(payload=_payload(1))], "next"
        return [SimpleNamespace(payload=_payload(0))], None


class _Retriever:
    async def search(self, query, *, top_k, paper_ids):
        self.query = query
        self.top_k = top_k
        self.paper_ids = paper_ids
        return [
            Document(
                page_content="The paper reports accuracy.",
                metadata={
                    "paper_id": "120",
                    "chunk_id": "chunk-1",
                    "chunk_index": 1,
                    "page_numbers": [4],
                    "source_spans": _payload(1)["source_spans"],
                },
            )
        ]


class PaperContentEvaluationAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_reader_paginates_and_sorts_qdrant_payloads(self) -> None:
        reader = PaperContentCorpusReader(
            collection_name="paper-content-test",
            client=_ScrollClient(),
        )

        documents = await reader.read(["120"])

        self.assertEqual(
            [document.metadata["chunk_index"] for document in documents],
            [0, 1],
        )
        self.assertEqual(reader.page_aware_documents(documents), documents)
        self.assertEqual(len(reader.fingerprint(documents)), 64)

    async def test_target_delegates_to_the_existing_content_retriever(self) -> None:
        retriever = _Retriever()
        target = PaperContentRetrievalTarget(retriever=retriever)
        case = EvaluationCase(
            case_id="case-1",
            paper_id="120",
            query="Which metric is reported?",
            reference_answer="Accuracy.",
            scenario="single_hop",
            evidence=[
                EvidenceRef(
                    paper_id="120",
                    chunk_id_hint="chunk-1",
                    page_numbers=[4],
                    supporting_quote="accuracy",
                    quote_sha256="a" * 64,
                    chunk_char_start=19,
                    chunk_char_end=27,
                )
            ],
        )

        observation = await target.evaluate(case, top_k=7)

        self.assertEqual(retriever.query, case.query)
        self.assertEqual(retriever.top_k, 7)
        self.assertEqual(retriever.paper_ids, ["120"])
        self.assertEqual(observation.retrieved_contexts[0].page_numbers, [4])


if __name__ == "__main__":
    unittest.main()
