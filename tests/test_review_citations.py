import json
import tempfile
import unittest

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.nodes.assemble_review import (
    AssembleReviewNode,
)
from app.llm.graph.workflows.review_generate.nodes.finalizing_handoff import (
    FinalizingHandoffNode,
)
from app.llm.graph.workflows.review_generate.nodes.reflect_review import (
    ReflectReviewNode,
)


class _Model:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.messages = []

    async def ainvoke(self, messages):
        self.messages.append(messages)
        return self.responses.pop(0)


FRAMEWORK = {
    "title": "Review title",
    "scope": "A sufficiently detailed review scope.",
    "sections": [
        {
            "section_id": section_id,
            "title": section_id.title(),
            "description": "A sufficiently detailed section description.",
            "discussion_questions": ["What does the evidence show?"],
            "relevant_paper_ids": ["1", "2"],
        }
        for section_id in ("evidence", "methods", "findings", "synthesis")
    ],
}


class ReviewCitationTests(unittest.IsolatedAsyncioTestCase):
    async def test_assembly_repairs_synthesis_citations_outside_body(self) -> None:
        model = _Model(
            [
                {
                    "abstract": "Summary [[REF_2]]",
                    "conclusion": "Conclusion [[REF_2]]",
                },
                {
                    "abstract": "Summary [[REF_1]]",
                    "conclusion": "Conclusion [[REF_1]]",
                },
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            framework = await store.write_json(
                run_id="review-test",
                step_key="framework",
                source="test",
                kind="framework",
                payload={"framework": FRAMEWORK},
            )
            corpus = await store.write_json(
                run_id="review-test",
                step_key="corpus",
                source="test",
                kind="corpus",
                payload={
                    "papers": [
                        {"paper_id": 1, "title": "One"},
                        {"paper_id": 2, "title": "Two"},
                    ]
                },
            )
            section = await store.write_json(
                run_id="review-test",
                step_key="section",
                source="test",
                kind="section",
                payload={
                    "section_id": "evidence",
                    "summary": "Evidence summary.",
                    "text": "Evidence-backed text [[REF_1]].",
                },
            )

            update = await AssembleReviewNode(
                artifact_store=store,
                model=model,
            )(
                {
                    "run_id": "review-test",
                    "task_id": 1,
                    "language": "en",
                    "framework_hash": "hash",
                    "framework_artifact_ref": framework.artifact_uri,
                    "corpus_artifact_ref": corpus.artifact_uri,
                    "section_draft_artifact_refs": {"evidence": section.artifact_uri},
                }
            )
            draft = await store.read_json_uri(update["review_draft_artifact_ref"])

        self.assertEqual(len(model.messages), 2)
        self.assertEqual(draft["citation_paper_ids"], ["1"])
        self.assertEqual(draft["sections"][0]["discussion_questions"], ["What does the evidence show?"])
        self.assertNotIn("[[REF_2]]", draft["abstract"] + draft["conclusion"])

    def test_reflection_flags_synthesis_citations_outside_body(self) -> None:
        issues = ReflectReviewNode()._check_hard_constraints(
            review_draft={
                "abstract": "Summary [[REF_2]]",
                "body_markdown": "Evidence [[REF_1]].",
                "conclusion": "Conclusion [[REF_1]]",
                "sections": [{"section_id": "evidence"}],
            },
            corpus_payload={"papers": [{"paper_id": 1}, {"paper_id": 2}]},
            framework={**FRAMEWORK, "sections": [FRAMEWORK["sections"][0]]},
        )

        self.assertIn(
            "synthesis cites a paper outside the evidence-backed body: REF_2",
            issues,
        )

    async def test_stale_reflection_report_cannot_approve_a_new_draft(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            draft = await store.write_json(
                run_id="review-test",
                step_key="draft",
                source="test",
                kind="draft",
                payload={},
            )
            report = await store.write_json(
                run_id="review-test",
                step_key="report",
                source="test",
                kind="report",
                payload={
                    "effective_decision": "pass",
                    "hard_issues": [],
                    "review_draft_artifact_ref": "artifact://old-draft",
                },
            )
            corpus = await store.write_json(
                run_id="review-test",
                step_key="corpus",
                source="test",
                kind="corpus",
                payload={"papers": []},
            )

            update = await FinalizingHandoffNode(artifact_store=store)(
                {
                    "reflection_report_artifact_ref": report.artifact_uri,
                    "review_draft_artifact_ref": draft.artifact_uri,
                    "corpus_artifact_ref": corpus.artifact_uri,
                }
            )

        self.assertEqual(update["stage"], "failed")


if __name__ == "__main__":
    unittest.main()
