from __future__ import annotations

import unittest

from app.rag.evaluation.runner import RagEvaluationRunner, render_markdown_report
from app.rag.evaluation.schemas import (
    EvaluationCase,
    EvaluationObservation,
    EvidenceRef,
    RetrievedContext,
)


class _Target:
    name = "fake"

    async def evaluate(self, case, *, top_k):
        return EvaluationObservation(
            case_id=case.case_id,
            latency_ms=12.5,
            retrieved_contexts=[
                RetrievedContext(
                    rank=1,
                    paper_id=case.paper_id,
                    chunk_id="chunk-1",
                    content="The paper reports accuracy.",
                    page_numbers=[4],
                )
            ],
        )


class RagEvaluationRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_reports_deterministic_retrieval_metrics(self) -> None:
        case = EvaluationCase(
            case_id="case-1",
            paper_id="120",
            query="Which metric is reported?",
            reference_answer="The paper reports accuracy.",
            scenario="single_hop",
            evidence=[
                EvidenceRef(
                    paper_id="120",
                    chunk_id_hint="chunk-1",
                    page_numbers=[4],
                    supporting_quote="The paper reports accuracy.",
                    quote_sha256="a" * 64,
                    chunk_char_start=0,
                    chunk_char_end=27,
                )
            ],
        )

        result = await RagEvaluationRunner(_Target()).run(
            dataset_id="paper-content-v1",
            cases=[case],
            top_k=20,
            metrics={"deterministic"},
        )

        self.assertEqual(result.summary["quote_recall"], 1.0)
        self.assertEqual(result.summary["page_recall"], 1.0)
        self.assertEqual(result.summary["evidence_mrr"], 1.0)
        self.assertIn("# RAG evaluation", render_markdown_report(result))


if __name__ == "__main__":
    unittest.main()
