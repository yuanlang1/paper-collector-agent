import json
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.nodes.assemble_review import (
    AssembleReviewNode,
    ReviewSynthesis,
)
from app.llm.graph.workflows.review_generate.nodes.finalizing_handoff import (
    FinalizingHandoffNode,
)
from app.llm.graph.workflows.review_generate.nodes.persist_review import PersistReviewNode
from app.llm.graph.workflows.review_generate.nodes.finalize import (
    finalize_task_review_node,
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
        return ReviewSynthesis.model_validate(self.responses.pop(0))


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
    async def test_assembly_inserts_synthesis_citations_from_selected_sources(self) -> None:
        model = _Model(
            [
                {
                    "abstract": [{"text": "Summary", "source_indices": [1]}],
                    "conclusion": [{"text": "Conclusion", "source_indices": [1]}],
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
                    "text": "Evidence-backed text [@paper-1].",
                },
            )

            update = await AssembleReviewNode(
                artifact_store=store,
                model=model,
            )(
                {
                    "run_id": "review-test",
                    "child_run_id": "child-review-test",
                    "task_id": 1,
                    "language": "en",
                    "framework_hash": "hash",
                    "framework_artifact_ref": framework.artifact_uri,
                    "corpus_artifact_ref": corpus.artifact_uri,
                    "section_draft_artifact_refs": {"evidence": section.artifact_uri},
                }
            )
            draft = await store.read_json_uri(update["review_draft_artifact_ref"])

        self.assertEqual(len(model.messages), 1)
        self.assertEqual(draft["citation_paper_ids"], ["1"])
        self.assertEqual(draft["sections"][0]["discussion_questions"], ["What does the evidence show?"])
        self.assertEqual(draft["abstract"], "Summary [@paper-1]")
        self.assertEqual(draft["conclusion"], "Conclusion [@paper-1]")

    def test_reflection_flags_synthesis_citations_outside_body(self) -> None:
        issues = ReflectReviewNode()._check_hard_constraints(
            review_draft={
                "abstract": "Summary [@paper-2]",
                "body_markdown": "Evidence [@paper-1].",
                "conclusion": "Conclusion [@paper-1]",
                "sections": [{"section_id": "evidence"}],
            },
            corpus_payload={"papers": [{"paper_id": 1}, {"paper_id": 2}]},
            framework={**FRAMEWORK, "sections": [FRAMEWORK["sections"][0]]},
        )

        self.assertIn(
            "synthesis cites a paper outside the evidence-backed body: paper-2",
            issues,
        )

    def test_final_handoff_keeps_pandoc_citation_keys(self) -> None:
        review = FinalizingHandoffNode()._build_final_review(
            review_draft={
                "task_id": 1,
                "title": "Review",
                "scope": "Scope",
                "abstract": "Abstract [@paper-1]",
                "sections": [],
                "body_markdown": "Body [@paper-1; @paper-2]",
                "conclusion": "Conclusion [@paper-2]",
            },
            corpus={
                "papers": [
                    {
                        "paper_id": 1,
                        "title": "One",
                        "authors": [],
                        "published_date": None,
                        "doi": None,
                    },
                    {
                        "paper_id": 2,
                        "title": "Two",
                        "authors": [],
                        "published_date": None,
                        "doi": None,
                    },
                ]
            },
            citation_style="harvard",
            reflection_report_artifact_ref="artifact://review-test/report/report.json",
        )

        self.assertEqual(review["citation_labels"], {"1": "paper-1", "2": "paper-2"})
        self.assertEqual(review["citation_paper_ids"], ["1", "2"])
        self.assertIn("[@paper-1; @paper-2]", review["markdown"])
        self.assertNotIn("## References", review["markdown"])

    async def test_persistence_uses_the_service_version_number(self) -> None:
        client = Mock()
        client.add_review = AsyncMock(
            return_value={
                "ok": True,
                "result": {"review_id": 9, "version_number": 4},
                "error": None,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            final = await store.write_json(
                run_id="review-test",
                step_key="final",
                source="test",
                kind="final",
                payload={
                    "title": "Review",
                    "scope": "Scope",
                    "abstract": "Abstract",
                    "body_markdown": "Body [@paper-1]",
                    "conclusion": "Conclusion",
                    "markdown": "# Review\n\nBody [@paper-1]",
                    "sections": [],
                    "citation_paper_ids": ["1"],
                    "citation_labels": {"1": "paper-1"},
                    "references": [{"paper_id": "1", "formatted": "One"}],
                },
            )
            update = await PersistReviewNode(
                artifact_store=store, client=client
            )(
                {
                    "stage": "persisting_review",
                    "final_review_artifact_ref": final.artifact_uri,
                    "task_id": 1,
                    "topic": "Topic",
                    "review_version_number": 1,
                    "language": "en",
                    "review_type": "narrative",
                    "citation_style": "harvard",
                    "paper_ids_snapshot": ["1"],
                    "framework_hash": "hash",
                }
            )

        self.assertEqual(update["review_id"], 9)
        self.assertEqual(update["review_version_number"], 4)

    async def test_failed_finalization_keeps_completed_verification(self) -> None:
        store = AsyncMock()
        with patch(
            "app.llm.graph.workflows.review_generate.nodes.finalize.LocalArtifactStore",
            return_value=store,
        ):
            result = await finalize_task_review_node(
                {
                    "active_tool_call": {"id": "review", "name": "task_review_agent"},
                    "child_run_id": "child-review-finalize",
                    "stage": "failed",
                    "error_code": "VERIFICATION_PROTOCOL_FAILED",
                    "error": "invalid source index",
                    "claim_verification_artifact_ref": "artifact://review/verify/result.json",
                }
            )

        store.delete_run_directories.assert_not_awaited()
        self.assertEqual(
            result["last_action_result"]["artifact_refs"],
            ["artifact://review/verify/result.json"],
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
