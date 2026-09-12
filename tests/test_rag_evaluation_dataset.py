from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.rag.evaluation.dataset import EvaluationDatasetStore
from app.rag.evaluation.schemas import (
    DatasetManifest,
    EvaluationCase,
    EvidenceRef,
)


def _case(case_id: str = "case-1") -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        paper_id="120",
        query="What is evaluated?",
        reference_answer="Accuracy is evaluated.",
        scenario="single_hop",
        evidence=[
            EvidenceRef(
                paper_id="120",
                chunk_id_hint="chunk-1",
                page_numbers=[4],
                supporting_quote="Accuracy is evaluated.",
                quote_sha256="a" * 64,
                chunk_char_start=0,
                chunk_char_end=22,
                block_types=["text"],
            )
        ],
    )


class EvaluationDatasetStoreTests(unittest.TestCase):
    def test_review_and_publish_only_approved_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EvaluationDatasetStore(Path(directory))
            store.write_draft("paper-content-v1", [_case("case-1"), _case("case-2")])
            store.update_review(
                "paper-content-v1",
                case_ids={"case-1"},
                status="approved",
                reviewer="reviewer",
                note="verified page",
            )

            store.publish("paper-content-v1")

            published = store.load_published("paper-content-v1")
            self.assertEqual([case.case_id for case in published], ["case-1"])
            self.assertEqual(published[0].review.status, "approved")

    def test_manifest_and_dataset_fingerprint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EvaluationDatasetStore(Path(directory))
            cases = [_case()]
            fingerprint = store.dataset_fingerprint(cases)
            manifest = DatasetManifest(
                dataset_id="paper-content-v1",
                collection_name="paper-content",
                paper_ids=["120"],
                corpus_fingerprint="b" * 64,
            )

            store.write_manifest(manifest)

            self.assertEqual(store.load_manifest(manifest.dataset_id), manifest)
            self.assertEqual(fingerprint, store.dataset_fingerprint(cases))


if __name__ == "__main__":
    unittest.main()
