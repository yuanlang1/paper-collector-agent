from __future__ import annotations

import unittest

from langchain_core.documents import Document

from app.rag.evaluation.generation import (
    DatasetGenerator,
    GeneratedEvaluationCandidate,
    GeneratedEvidence,
)


class _CandidateModel:
    async def ainvoke(self, _messages):
        return GeneratedEvaluationCandidate(
            query="Which metric is reported?",
            reference_answer="The paper reports accuracy.",
            scenario="single_hop",
            evidence=[
                GeneratedEvidence(
                    evidence_key="evidence-1",
                    supporting_quote="The paper reports accuracy.",
                )
            ],
        )


class DatasetGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_generated_quote_maps_to_its_exact_page(self) -> None:
        content = "Heading\n\nThe paper reports accuracy."
        document = Document(
            page_content=content,
            metadata={
                "paper_id": "120",
                "chunk_id": "chunk-1",
                "chunk_index": 0,
                "page_numbers": [3, 4],
                "source_spans": [
                    {
                        "page": 3,
                        "type": "title",
                        "source_index": 0,
                        "char_start": 0,
                        "char_end": 7,
                    },
                    {
                        "page": 4,
                        "type": "text",
                        "source_index": 1,
                        "char_start": 9,
                        "char_end": len(content),
                    },
                ],
            },
        )
        generator = DatasetGenerator(_CandidateModel())

        cases = await generator.generate(
            [document],
            candidate_count=1,
            scenarios=["single_hop"],
        )

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].evidence[0].page_numbers, [4])
        self.assertEqual(cases[0].evidence[0].block_types, ["text"])

    async def test_interleaves_requested_scenarios(self) -> None:
        documents = [
            Document(
                page_content=f"content {page}",
                metadata={
                    "paper_id": "120",
                    "chunk_id": f"chunk-{page}",
                    "chunk_index": page,
                    "page_numbers": [page],
                    "source_spans": [
                        {
                            "page": page,
                            "type": "text",
                            "source_index": page,
                            "char_start": 0,
                            "char_end": len(f"content {page}"),
                        }
                    ],
                },
            )
            for page in (1, 2, 3)
        ]

        bundles = DatasetGenerator._build_bundles(
            documents,
            ["single_hop", "multi_hop"],
        )

        self.assertEqual(
            [bundle.scenario for bundle in bundles[:4]],
            ["single_hop", "multi_hop", "single_hop", "multi_hop"],
        )


if __name__ == "__main__":
    unittest.main()
