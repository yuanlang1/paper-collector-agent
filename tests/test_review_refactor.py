import copy
import json
import tempfile
import unittest
from unittest.mock import AsyncMock

from langchain_core.documents import Document

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.contracts import (
    claim_hash,
    save,
    validate_claims,
)
from app.llm.graph.workflows.review_generate.nodes.generate_claim import GenerateClaimsNode
from app.llm.graph.workflows.review_generate.nodes.generate_framework import GenerateFrameworkNode
from app.llm.graph.workflows.review_generate.nodes.reflect_review import ReflectReviewNode
from app.llm.graph.workflows.review_generate.nodes.render_section import RenderSectionsNode
from app.llm.graph.workflows.review_generate.nodes.retrieve_evidence import RetrieveEvidenceNode
from app.llm.graph.workflows.review_generate.nodes.verify_claims import VerifyClaimsNode
from app.llm.graph.workflows.review_generate.revisions import schedule_revision
from app.llm.graph.workflows.review_generate.workflow import build_task_review_workflow


class Model:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.inputs = []

    async def ainvoke(self, messages):
        self.inputs.append(json.loads(messages[1].content))
        return copy.deepcopy(self.responses.pop(0))


def document(paper_id="1", chunk_id="chunk1", text="The result is conditional."):
    return Document(
        page_content=text,
        metadata={
            "paper_id": paper_id,
            "chunk_id": chunk_id,
            "chunk_index": 0,
            "section_path": "Results",
            "page_numbers": [3],
        },
    )


FRAMEWORK = {
    "title": "Conditional outcomes",
    "scope": "A fixed corpus synthesis of conditional outcomes.",
    "sections": [
        {
            "section_id": "results",
            "title": "Results",
            "description": "How do reported study conditions affect outcomes?",
            "relevant_paper_ids": ["1", "2"],
        }
    ],
    "excluded_papers": [],
}
CANDIDATE = {
    "text": "The results depend on reported study conditions.",
    "candidate_paper_ids": ["1", "2"],
    "retrieval_queries": ["conditional outcomes"],
    "claim_type": "comparative",
    "evidence_requirement": "multiple_fulltext",
    "withdrawn_reason": "",
}
CLAIM = dict(CANDIDATE, claim_id="claim_001", section_id="results")
CLAIM["claim_hash"] = claim_hash(CLAIM)


def revision(kind="retrieval", target="claim_001"):
    return {
        "target_type": kind,
        "target_id": target,
        "issue": "Missing comparison",
        "required_change": "Find conditional outcomes",
        "acceptance_criteria": "Comparable evidence",
        "queries": ["conditional outcomes"],
    }


def verdict(status="supported"):
    return {
        "status": status,
        "evidence": [
            {
                "paper_id": paper_id,
                "chunk_id": f"chunk{paper_id}",
                "quote": "The result is conditional.",
                "relation": "supports",
            }
            for paper_id in ("1", "2")
        ],
        "reason": "Both original texts support conditionality",
        "revisions": [] if status == "supported" else [revision()],
    }


class ReviewRefactorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = LocalArtifactStore(self.directory.name)
        self.state = {
            "run_id": "review-test",
            "task_id": 1,
            "topic": "Conditional outcomes",
            "language": "en",
            "review_type": "narrative",
            "paper_ids_snapshot": ["1", "2"],
            "max_reflection_rounds": 3,
            "reflection_round": 0,
            "revision_items": [],
            "warnings": [],
        }
        self.state["study_records_artifact_ref"] = await save(
            self.store,
            self.state,
            "studies",
            {
                "studies": [
                    {
                        "paper_id": paper_id,
                        "core_problem": "Problem",
                        "methods": "Method",
                        "main_discussion": "Conditional result",
                    }
                    for paper_id in ("1", "2")
                ]
            },
        )

    async def seed_framework(self, framework=FRAMEWORK, changed_section_ids=None):
        self.state["framework_artifact_ref"] = await save(
            self.store,
            self.state,
            "framework",
            {
                "framework": framework,
                "framework_hash": "framework-hash",
                "framework_input_hash": "input-hash",
                "changed_section_ids": changed_section_ids or [],
            },
        )
        self.state["framework_hash"] = "framework-hash"

    async def seed_claim(self):
        await self.seed_framework()
        self.state["claims_artifact_ref"] = await save(
            self.store,
            self.state,
            "claims",
            {"claims": [CLAIM], "unanswered": []},
        )

    async def retrieve(self, docs=None):
        retriever = AsyncMock()
        retriever.search.return_value = docs or [document(), document("2", "chunk2")]
        update = await RetrieveEvidenceNode(artifact_store=self.store, content_retrieval=retriever)(
            self.state
        )
        self.assertEqual(update["stage"], "verifying_claims", update)
        self.state.update(update)
        return retriever

    async def verify(self, result=None):
        update = await VerifyClaimsNode(
            artifact_store=self.store, model=Model(result or verdict())
        )(self.state)
        self.state.update(update)
        return update

    async def test_framework_uses_study_profiles_before_claims(self):
        model = Model(FRAMEWORK)
        update = await GenerateFrameworkNode(artifact_store=self.store, model=model)(self.state)
        self.state.update(update)
        self.assertEqual(update["stage"], "generating_claims")
        self.assertEqual(model.inputs[0]["overview"][0]["main_discussion"], "Conditional result")

        claims = await GenerateClaimsNode(
            artifact_store=self.store, model=Model({"claims": [CANDIDATE]})
        )(self.state)
        payload = await self.store.read_json_uri(claims["claims_artifact_ref"])
        self.assertEqual(payload["claims"][0]["section_id"], "results")
        self.assertNotIn("question_id", payload["claims"][0])
        self.assertEqual(claims["changed_section_ids"], [])

    async def test_framework_cache_does_not_depend_on_claim_changes(self):
        first = Model(FRAMEWORK)
        self.state.update(await GenerateFrameworkNode(artifact_store=self.store, model=first)(self.state))
        cached = Model()
        update = await GenerateFrameworkNode(artifact_store=self.store, model=cached)(self.state)
        self.assertEqual(update["stage"], "generating_claims")
        self.assertEqual(cached.inputs, [])

    async def test_claim_invariants_use_sections(self):
        await self.seed_framework()
        for claims in (
            [CLAIM, CLAIM],
            [dict(CLAIM, candidate_paper_ids=["99"])],
            [dict(CLAIM, section_id="missing")],
        ):
            with self.assertRaises(ValueError):
                validate_claims(claims, FRAMEWORK, ["1", "2"])

    async def test_claim_split_inherits_section_and_reenters_verification(self):
        await self.seed_claim()
        await self.retrieve()
        self.state["revision_items"] = [revision("claim")]
        children = [
            dict(CANDIDATE, text="Results depend on sample size."),
            dict(CANDIDATE, text="Results depend on intervention duration."),
        ]
        update = await GenerateClaimsNode(
            artifact_store=self.store, model=Model(), revision_model=Model({"claims": children})
        )(self.state)
        payload = await self.store.read_json_uri(update["claims_artifact_ref"])
        self.assertEqual([claim["claim_id"] for claim in payload["claims"]], ["claim_001_1", "claim_001_2"])
        self.assertEqual({claim["section_id"] for claim in payload["claims"]}, {"results"})
        self.state.update(update)
        await self.retrieve()
        result = await VerifyClaimsNode(
            artifact_store=self.store, model=Model(verdict(), verdict())
        )(self.state)
        self.assertEqual(result["stage"], "rendering_sections")

    async def test_add_claim_targets_a_section(self):
        await self.seed_claim()
        self.state["revision_items"] = [revision("add_claim", "results")]
        update = await GenerateClaimsNode(
            artifact_store=self.store,
            model=Model({"claims": [dict(CANDIDATE, text="Noise can reverse the reported outcome.")]}),
        )(self.state)
        payload = await self.store.read_json_uri(update["claims_artifact_ref"])
        self.assertEqual([claim["claim_id"] for claim in payload["claims"]], ["claim_001", "claim_002"])
        self.assertEqual(payload["claims"][1]["section_id"], "results")

    async def test_supported_comparison_with_one_source_reenters_retrieval(self):
        await self.seed_claim()
        await self.retrieve()
        result = verdict()
        result["evidence"] = result["evidence"][:1]
        update = await self.verify(result)
        self.assertEqual(update["stage"], "retrieving_evidence")
        self.assertEqual(update["revision_items"][0]["target_type"], "retrieval")

    async def test_no_claims_continue_to_material_insufficiency_sections(self):
        await self.seed_framework()
        self.state["claims_artifact_ref"] = await save(
            self.store, self.state, "claims", {"claims": [], "unanswered": []}
        )
        await self.retrieve()
        update = await VerifyClaimsNode(artifact_store=self.store, model=Model())(self.state)
        self.assertEqual(update["stage"], "rendering_sections")
        self.assertIn("material insufficiency", update["warnings"][-1])

    async def test_exhausted_evidence_revision_renders_without_the_claim(self):
        await self.seed_claim()
        await self.retrieve()
        self.state["reflection_round"] = self.state["max_reflection_rounds"] - 1
        update = await self.verify(verdict("insufficient"))
        self.assertEqual(update["stage"], "rendering_sections")
        self.assertEqual(update["revision_items"], [])

    async def test_shared_quote_uses_the_current_claim_relation(self):
        framework = copy.deepcopy(FRAMEWORK)
        await self.seed_framework(framework)
        first = dict(CLAIM, claim_id="claim_a", claim_type="descriptive", evidence_requirement="fulltext")
        first["claim_hash"] = claim_hash(first)
        second = dict(CLAIM, claim_id="claim_b", claim_type="descriptive", evidence_requirement="fulltext")
        second["claim_hash"] = claim_hash(second)
        self.state["claims_artifact_ref"] = await save(
            self.store, self.state, "claims", {"claims": [first, second], "unanswered": []}
        )
        shared = {
            "paper_id": "1",
            "chunk_id": "shared",
            "quote": "Shared result.",
            "text": "Shared result.",
            "evidence_id": "shared",
        }
        self.state["claim_verification_artifact_ref"] = await save(
            self.store,
            self.state,
            "verification",
            {
                "claims": [
                    {
                        "claim_id": "claim_a",
                        "claim_hash": first["claim_hash"],
                        "status": "supported",
                        "reason": "",
                        "evidence": [dict(shared, relation="opposes"), dict(shared, evidence_id="a_support", paper_id="2", relation="supports")],
                    },
                    {
                        "claim_id": "claim_b",
                        "claim_hash": second["claim_hash"],
                        "status": "supported",
                        "reason": "",
                        "evidence": [dict(shared, relation="supports")],
                    },
                ]
            },
        )
        model = Model(
            {
                "arguments": [
                    {
                        "text": "Shared result [[REF_1]].",
                        "claim_ids": ["claim_a"],
                        "evidence_ids": ["shared"],
                    },
                    {
                        "text": "Shared result [[REF_1]].",
                        "claim_ids": ["claim_b"],
                        "evidence_ids": ["shared"],
                    },
                ],
                "summary": "Summary",
            }
        )
        update = await RenderSectionsNode(artifact_store=self.store, model=model)(self.state)
        self.assertEqual(update["error_code"], "RENDER_FAILED")

    async def test_section_without_supported_claims_writes_only_a_notice(self):
        await self.seed_framework()
        self.state["claims_artifact_ref"] = await save(
            self.store, self.state, "claims", {"claims": [CLAIM], "unanswered": []}
        )
        self.state["claim_verification_artifact_ref"] = await save(
            self.store,
            self.state,
            "verification",
            {"claims": [dict(verdict("insufficient"), claim_id="claim_001", claim_hash=CLAIM["claim_hash"])]},
        )
        update = await RenderSectionsNode(
            artifact_store=self.store,
            model=Model({"arguments": [], "summary": "Insufficient", "insufficient_notice": "The current material is insufficient."}),
        )(self.state)
        draft = await self.store.read_json_uri(update["section_draft_artifact_refs"]["results"])
        self.assertEqual(draft["used_claim_ids"], [])
        self.assertEqual(draft["text"], "The current material is insufficient.")

    async def test_framework_revision_marks_only_changed_sections(self):
        await self.seed_claim()
        changed = copy.deepcopy(FRAMEWORK)
        changed["sections"][0]["description"] = "Which conditions change outcomes?"
        framework_revision = revision("framework", "review")
        claim_revision = revision("claim", "claim_001")
        self.state["revision_items"] = [framework_revision, claim_revision]
        model = Model(changed)
        update = await GenerateFrameworkNode(artifact_store=self.store, model=model)(self.state)
        payload = await self.store.read_json_uri(update["framework_artifact_ref"])
        self.assertEqual(payload["changed_section_ids"], ["results"])
        self.assertNotIn(claim_revision, update["revision_items"])
        self.assertIn(framework_revision, model.inputs[0]["revisions"])

    async def test_framework_replan_remaps_pending_section_revisions(self):
        await self.seed_claim()
        changed = copy.deepcopy(FRAMEWORK)
        changed["sections"][0]["section_id"] = "conditions"
        changed["section_revision_targets"] = {"results": "conditions"}
        framework_revision = revision("framework", "review")
        section_revision = revision("section", "results")
        add_claim_revision = revision("add_claim", "results")
        self.state["revision_items"] = [
            framework_revision,
            section_revision,
            add_claim_revision,
        ]
        model = Model(changed)
        update = await GenerateFrameworkNode(artifact_store=self.store, model=model)(self.state)
        routed = {
            item["target_type"]: item["target_id"] for item in update["revision_items"]
        }
        self.assertEqual(routed, {"section": "conditions", "add_claim": "conditions"})
        self.assertIn(section_revision, model.inputs[0]["revisions"])
        self.assertIn(add_claim_revision, model.inputs[0]["revisions"])

    async def test_framework_replan_consumes_a_handled_section_revision(self):
        await self.seed_claim()
        changed = copy.deepcopy(FRAMEWORK)
        changed["sections"][0]["section_id"] = "conditions"
        changed["section_revision_targets"] = {"results": None}
        self.state["revision_items"] = [
            revision("framework", "review"),
            revision("section", "results"),
        ]
        update = await GenerateFrameworkNode(
            artifact_store=self.store, model=Model(changed)
        )(self.state)
        self.assertEqual(update["revision_items"], [])

    async def test_schedule_accepts_add_claim(self):
        result = await schedule_revision(
            self.store,
            self.state,
            [revision("add_claim", "results")],
            claim_ids=[],
            section_ids=["results"],
        )
        self.assertEqual(result["stage"], "generating_claims")

    def test_reflection_splits_large_evidence_without_dropping_relations(self):
        checks = ReflectReviewNode()._claim_checks(
            CLAIM,
            {
                "status": "supported",
                "reason": "",
                "evidence": [
                    {"evidence_id": "support", "paper_id": "1", "chunk_id": "a", "relation": "supports", "text": "a" * 14000},
                    {"evidence_id": "oppose", "paper_id": "2", "chunk_id": "b", "relation": "opposes", "text": "b" * 14000},
                ],
            },
            [{"text": "body" * 2000, "claim_ids": ["claim_001"], "evidence_ids": ["support"]}],
        )
        self.assertGreater(len(checks), 1)
        self.assertEqual({item["relation"] for check in checks for item in check["verification"]["evidence"]}, {"supports", "opposes"})

    def test_reflection_allows_an_uncited_material_insufficiency_review(self):
        issues = ReflectReviewNode()._check_hard_constraints(
            review_draft={
                "body_markdown": "Current material is insufficient.",
                "abstract": "Current material is insufficient.",
                "conclusion": "Current material is insufficient.",
            },
            corpus_payload={"papers": []},
            require_citations=False,
        )
        self.assertNotIn("review has no citation anchors", issues)

    async def test_workflow_orders_framework_before_claims(self):
        order = []
        transitions = {
            "initialize": "loading_corpus",
            "load_task_corpus": "extracting_studies",
            "extract_studies": "generating_framework",
            "generate_framework": "generating_claims",
            "generate_claims": "retrieving_evidence",
            "retrieve_evidence": "verifying_claims",
            "verify_claims": "rendering_sections",
            "render_sections": "assembling_review",
            "assemble_review": "reflecting_review",
            "reflect_review": "finalizing_handoff",
            "finalizing_handoff": "persisting_review",
            "persist_review": "completed",
            "finalize_result": "completed",
        }

        def node(name):
            async def run(state):
                order.append(name)
                return {"stage": transitions[name]}

            return run

        graph = build_task_review_workflow(
            node_overrides={name: node(name) for name in transitions}
        )
        result = await graph.ainvoke({})
        self.assertEqual(result["stage"], "completed")
        self.assertLess(order.index("generate_framework"), order.index("generate_claims"))
        self.assertNotIn("resolve_review_focus", order)


if __name__ == "__main__":
    unittest.main()
