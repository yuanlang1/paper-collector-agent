import copy
import json
import re
import tempfile
import unittest
from unittest.mock import AsyncMock

from langchain_core.documents import Document

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.contracts import (
    CandidateClaim,
    claim_hash,
    save,
    validate_claims,
)
from app.llm.graph.workflows.review_generate.nodes.assemble_review import (
    ASSEMBLE_REVIEW_PROMPT,
    REPAIR_SYNTHESIS_PROMPT,
    ReviewSynthesis,
)
from app.llm.graph.workflows.review_generate.nodes.extract_studies import (
    ARTICLE_PROFILE_PROMPT,
    ARTICLE_PROFILE_TOOLS,
    PROFILE_SUBMISSION_PROMPT,
    ProfileField,
    SearchCurrentPaperArgs,
    SubmittedArticleProfile,
)
from app.llm.graph.workflows.review_generate.nodes.generate_claim import (
    CLAIM_GENERATION_PROMPT,
    CLAIM_REVISION_PROMPT,
    ClaimRevisionPlan,
    ClaimsPlan,
    GenerateClaimsNode,
)
from app.llm.graph.workflows.review_generate.nodes.generate_framework import (
    FRAMEWORK_PROMPT,
    ExcludedPaper,
    FrameworkSection,
    GenerateFrameworkNode,
    ReviewFramework,
)
from app.llm.graph.workflows.review_generate.nodes.initialize import initialize_review_node
from app.llm.graph.workflows.review_generate.nodes.reflect_review import (
    ReflectReviewNode,
    WRITING_REVIEW_PROMPT,
    WritingReviewResult,
    WritingRevision,
)
from app.llm.graph.workflows.review_generate.nodes.render_section import (
    SECTION_DRAFT_PROMPT,
    Argument,
    RenderSectionsNode,
    SectionDraft,
)
from app.llm.graph.workflows.review_generate.nodes.retrieve_evidence import RetrieveEvidenceNode
from app.llm.graph.workflows.review_generate.nodes.verify_claims import (
    CLAIM_VERIFICATION_PROMPT,
    ClaimRevision,
    EvidenceQuote,
    Verdict,
    VerifyClaimsNode,
)
from app.llm.graph.workflows.review_generate.workflow import (
    _route_after_verification,
    _route_after_writing_review,
    build_task_review_workflow,
)
from app.llm.provider.structured_output import build_json_mode_instruction
from app.llm.subagents.task_review import TaskReviewDelegation


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
            "discussion_questions": ["How do study conditions affect outcomes?"],
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


def revision(kind="claim", target="claim_001"):
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


def writing_review(decision="pass", revisions=None, suggestions=None):
    return {
        "decision": decision,
        "summary": "Writing review completed.",
        "revisions": revisions or [],
        "suggestions": suggestions or [],
    }


def writing_revision(target_type="section", target_id="results"):
    return {
        "target_type": target_type,
        "target_id": target_id,
        "issue": "The conclusion is broader than the verified claim.",
        "required_change": "Restore the verified study conditions.",
        "acceptance_criteria": "The condition remains explicit in the draft.",
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
            "max_writing_revision_rounds": 1,
            "writing_revision_round": 0,
            "writing_review_plan_ref": None,
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

    async def seed_review_draft(self):
        await self.seed_claim()
        self.state["corpus_artifact_ref"] = await save(
            self.store,
            self.state,
            "corpus",
            {"papers": [{"paper_id": 1}, {"paper_id": 2}]},
        )
        self.state["claim_verification_artifact_ref"] = await save(
            self.store,
            self.state,
            "claim_verification",
            {
                "claims": [
                    {
                        "claim_id": CLAIM["claim_id"],
                        "claim_hash": CLAIM["claim_hash"],
                        "status": "supported",
                        "reason": "Supported.",
                        "evidence": [],
                    }
                ]
            },
        )
        self.state["review_draft_artifact_ref"] = await save(
            self.store,
            self.state,
            "review_draft",
            {
                "abstract": "Abstract [[REF_1]].",
                "sections": [
                    {
                        "section_id": "results",
                        "title": "Results",
                        "text": "The result depends on study conditions [[REF_1]] [[REF_2]].",
                        "summary": "Conditional outcomes.",
                        "arguments": [],
                        "used_claim_ids": ["claim_001"],
                        "omitted_claim_ids": [],
                    }
                ],
                "body_markdown": "## Results\n\nThe result depends on study conditions [[REF_1]] [[REF_2]].",
                "conclusion": "The outcome is conditional [[REF_1]].",
            },
        )

    async def renew_review_draft(self):
        draft = await self.store.read_json_uri(self.state["review_draft_artifact_ref"])
        self.state["review_draft_artifact_ref"] = await save(
            self.store, self.state, "review_draft", draft
        )

    async def test_framework_uses_study_profiles_before_claims(self):
        model = Model(FRAMEWORK)
        update = await GenerateFrameworkNode(artifact_store=self.store, model=model)(self.state)
        self.state.update(update)
        self.assertEqual(update["stage"], "generating_claims")
        self.assertEqual(model.inputs[0]["overview"][0]["main_discussion"], "Conditional result")
        framework = await self.store.read_json_uri(update["framework_artifact_ref"])
        self.assertEqual(
            framework["framework"]["sections"][0]["discussion_questions"],
            ["How do study conditions affect outcomes?"],
        )

        claim_model = Model({"claims": [CANDIDATE]})
        claims = await GenerateClaimsNode(
            artifact_store=self.store, model=claim_model
        )(self.state)
        payload = await self.store.read_json_uri(claims["claims_artifact_ref"])
        self.assertEqual(payload["claims"][0]["section_id"], "results")
        self.assertNotIn("question_id", payload["claims"][0])
        self.assertEqual(
            claim_model.inputs[0]["section"]["discussion_questions"],
            ["How do study conditions affect outcomes?"],
        )
        self.assertEqual(claims["changed_section_ids"], [])

    async def test_framework_requires_discussion_questions(self):
        framework = copy.deepcopy(FRAMEWORK)
        del framework["sections"][0]["discussion_questions"]

        update = await GenerateFrameworkNode(
            artifact_store=self.store, model=Model(framework)
        )(self.state)

        self.assertEqual(update["error_code"], "FRAMEWORK_FAILED")

    async def test_claim_generation_passes_all_section_studies_in_one_call(self):
        studies = [
            {"paper_id": paper_id, "main_discussion": "x" * 7000}
            for paper_id in ("1", "2")
        ]
        model = Model({"claims": []})
        claims, reason = await GenerateClaimsNode(model=model)._generate_for_section(
            self.state, FRAMEWORK["sections"][0], studies, [], []
        )

        self.assertEqual(claims, [])
        self.assertEqual(reason, "")
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(model.inputs[0]["studies"], studies)

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
        revision_model = Model({"claims": children})
        update = await GenerateClaimsNode(
            artifact_store=self.store, model=Model(), revision_model=revision_model
        )(self.state)
        payload = await self.store.read_json_uri(update["claims_artifact_ref"])
        self.assertEqual([claim["claim_id"] for claim in payload["claims"]], ["claim_001_1", "claim_001_2"])
        self.assertEqual({claim["section_id"] for claim in payload["claims"]}, {"results"})
        self.assertEqual(
            revision_model.inputs[0]["section"]["discussion_questions"],
            ["How do study conditions affect outcomes?"],
        )
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

    async def test_supported_comparison_with_one_source_revises_the_claim(self):
        await self.seed_claim()
        await self.retrieve()
        self.state["reflection_round"] = self.state["max_reflection_rounds"] - 2
        result = verdict()
        result["evidence"] = result["evidence"][:1]
        update = await self.verify(result)
        self.assertEqual(update["stage"], "generating_claims")
        self.assertEqual(update["revision_items"][0]["target_type"], "claim")
        self.assertEqual(update["reflection_round"], self.state["max_reflection_rounds"] - 1)

    async def test_exhausted_evidence_revision_renders_material_insufficiency(self):
        await self.seed_claim()
        await self.retrieve()
        self.state.update(max_reflection_rounds=1, reflection_round=0)

        update = await self.verify(verdict("insufficient"))

        self.assertEqual(update["stage"], "rendering_sections")
        self.assertEqual(update["revision_items"], [])
        self.assertIn("revision limit reached", update["warnings"][-1])
        verification = await self.store.read_json_uri(
            update["claim_verification_artifact_ref"]
        )
        self.assertEqual(verification["claims"][0]["status"], "insufficient")

    async def test_supported_claim_can_add_a_claim_in_its_section(self):
        await self.seed_claim()
        await self.retrieve()
        result = verdict()
        result["revisions"] = [revision("add_claim", "results")]
        model = Model(result, result)
        node = VerifyClaimsNode(artifact_store=self.store, model=model)

        first = await node(self.state)
        self.state.update(first)
        second = await node(self.state)

        self.assertEqual(first["stage"], "generating_claims")
        self.assertEqual(len(model.inputs), 2)
        self.assertEqual(second["error_code"], "REVIEW_NO_PROGRESS")
        self.assertEqual(model.inputs[0]["section"]["section_id"], "results")
        self.assertEqual(model.inputs[0]["section_claims"][0]["claim_id"], "claim_001")

    async def test_verify_rejects_invalid_revision_targets(self):
        await self.seed_claim()
        await self.retrieve()
        result = verdict("insufficient")
        result["revisions"] = [revision("claim", "other_claim")]
        update = await self.verify(result)
        self.assertEqual(update["error_code"], "VERIFICATION_FAILED")

        result = verdict()
        result["revisions"] = [revision("add_claim", "other_section")]
        update = await self.verify(result)
        self.assertEqual(update["error_code"], "VERIFICATION_FAILED")

    async def test_verify_rejects_non_claim_revision_types(self):
        await self.seed_claim()
        await self.retrieve()
        result = verdict("insufficient")
        result["revisions"] = [revision("framework", "review")]
        update = await self.verify(result)
        self.assertEqual(update["error_code"], "VERIFICATION_FAILED")

    async def test_verify_rejects_an_added_claim_without_a_claim_revision(self):
        await self.seed_claim()
        await self.retrieve()
        result = verdict("insufficient")
        result["revisions"] = [revision("add_claim", "results")]
        update = await self.verify(result)
        self.assertEqual(update["error_code"], "VERIFICATION_FAILED")

    async def test_retrieval_reuses_unchanged_claims_and_refreshes_changed_claims(self):
        await self.seed_claim()
        await self.retrieve()

        cached_retriever = AsyncMock()
        await RetrieveEvidenceNode(
            artifact_store=self.store, content_retrieval=cached_retriever
        )(self.state)
        cached_retriever.search.assert_not_awaited()

        changed = dict(CLAIM, text="The results depend on intervention duration.")
        changed["claim_hash"] = claim_hash(changed)
        self.state["claims_artifact_ref"] = await save(
            self.store, self.state, "claims", {"claims": [changed], "unanswered": []}
        )
        refreshed_retriever = AsyncMock()
        refreshed_retriever.search.return_value = [document(), document("2", "chunk2")]
        await RetrieveEvidenceNode(
            artifact_store=self.store, content_retrieval=refreshed_retriever
        )(self.state)
        refreshed_retriever.search.assert_awaited_once()

    async def test_retrieval_uses_each_claim_query_once(self):
        await self.seed_claim()
        claim = dict(
            CLAIM,
            candidate_paper_ids=["1"],
            retrieval_queries=["first query", "second query"],
        )
        self.state["claims_artifact_ref"] = await save(
            self.store, self.state, "claims", {"claims": [claim], "unanswered": []}
        )
        retriever = AsyncMock()
        retriever.search.return_value = [document(), document("2", "chunk2")]

        await RetrieveEvidenceNode(artifact_store=self.store, content_retrieval=retriever)(
            self.state
        )

        self.assertEqual(retriever.search.await_count, 2)
        self.assertEqual(
            [call.kwargs["query"] for call in retriever.search.await_args_list],
            ["first query", "second query"],
        )
        self.assertEqual(
            [call.kwargs["paper_ids"] for call in retriever.search.await_args_list],
            [["1"], ["1"]],
        )

    async def test_no_claims_continue_to_material_insufficiency_sections(self):
        await self.seed_framework()
        self.state["claims_artifact_ref"] = await save(
            self.store, self.state, "claims", {"claims": [], "unanswered": []}
        )
        await self.retrieve()
        update = await VerifyClaimsNode(artifact_store=self.store, model=Model())(self.state)
        self.assertEqual(update["stage"], "rendering_sections")
        self.assertIn("material insufficiency", update["warnings"][-1])

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

    async def test_section_writes_all_evidence_in_one_call(self):
        await self.seed_claim()
        evidence = [
            {
                "evidence_id": evidence_id,
                "paper_id": paper_id,
                "chunk_id": f"chunk_{paper_id}",
                "relation": "supports",
                "text": "e" * 25_000,
            }
            for evidence_id, paper_id in (("evidence_1", "1"), ("evidence_2", "2"))
        ]
        self.state["claim_verification_artifact_ref"] = await save(
            self.store,
            self.state,
            "verification",
            {
                "claims": [
                    {
                        "claim_id": CLAIM["claim_id"],
                        "claim_hash": CLAIM["claim_hash"],
                        "status": "supported",
                        "reason": "Supported.",
                        "evidence": evidence,
                    }
                ]
            },
        )
        model = Model(
            {
                "arguments": [
                    {
                        "text": "Conditional outcome [[REF_1]] [[REF_2]].",
                        "claim_ids": ["claim_001"],
                        "evidence_ids": ["evidence_1", "evidence_2"],
                    }
                ],
                "summary": "Conditional outcome.",
            }
        )

        update = await RenderSectionsNode(artifact_store=self.store, model=model)(self.state)

        self.assertEqual(update["stage"], "assembling_review")
        self.assertEqual(len(model.inputs), 1)

    async def test_writing_reflection_uses_one_compact_review_input(self):
        await self.seed_review_draft()
        model = Model(writing_review())

        update = await ReflectReviewNode(artifact_store=self.store, model=model)(self.state)

        self.assertEqual(update["stage"], "finalizing_handoff")
        self.assertEqual(len(model.inputs), 1)
        payload = model.inputs[0]
        self.assertEqual(
            payload["supported_claims"],
            [{"claim_id": "claim_001", "section_id": "results", "text": CLAIM["text"]}],
        )
        self.assertEqual(payload["review_mode"], "initial")
        self.assertEqual(
            payload["framework"]["sections"][0]["discussion_questions"],
            ["How do study conditions affect outcomes?"],
        )
        self.assertNotIn("study_profiles", payload)
        self.assertNotIn("evidence", payload)
        report = await self.store.read_json_uri(update["reflection_report_artifact_ref"])
        self.assertEqual(report["effective_decision"], "pass")
        self.assertEqual(report["hard_issues"], [])

    async def test_writing_reflection_saves_plan_for_targeted_revision(self):
        await self.seed_review_draft()
        revision_item = writing_revision()

        update = await ReflectReviewNode(
            artifact_store=self.store,
            model=Model(writing_review("revise", [revision_item])),
        )(self.state)

        self.assertEqual(update["stage"], "rendering_sections")
        self.assertEqual(update["writing_revision_round"], 1)
        self.assertEqual(update["revision_items"], [revision_item])
        plan = await self.store.read_json_uri(update["writing_review_plan_ref"])
        self.assertEqual(plan["review_draft_artifact_ref"], self.state["review_draft_artifact_ref"])
        self.assertEqual(plan["writing_revision_round"], 1)
        self.assertEqual(plan["revisions"], [revision_item])

    async def test_writing_reflection_stops_when_budget_is_exhausted(self):
        await self.seed_review_draft()
        self.state["max_writing_revision_rounds"] = 0

        update = await ReflectReviewNode(
            artifact_store=self.store,
            model=Model(writing_review("revise", [writing_revision()])),
        )(self.state)

        self.assertEqual(update["stage"], "failed")
        self.assertEqual(update["error_code"], "WRITING_REVISION_LIMIT_REACHED")
        report = await self.store.read_json_uri(update["reflection_report_artifact_ref"])
        self.assertEqual(report["stop_reason"], "writing_revision_limit_reached")

    async def test_writing_reflection_uses_two_revision_rounds_then_accepts(self):
        await self.seed_review_draft()
        self.state["max_writing_revision_rounds"] = 2
        model = Model(
            writing_review("revise", [writing_revision()]),
            writing_review("revise", [writing_revision()]),
            writing_review("pass"),
        )
        node = ReflectReviewNode(artifact_store=self.store, model=model)

        self.state.update(await node(self.state))
        self.state["revision_items"] = []
        await self.renew_review_draft()
        self.state.update(await node(self.state))
        self.state["revision_items"] = []
        await self.renew_review_draft()
        update = await node(self.state)

        self.assertEqual(update["stage"], "finalizing_handoff")
        self.assertEqual(self.state["writing_revision_round"], 2)
        self.assertEqual(len(model.inputs), 3)
        self.assertEqual(model.inputs[1]["review_mode"], "acceptance")
        self.assertIsNotNone(model.inputs[1]["previous_plan"])

    async def test_synthesis_revision_skips_section_rendering(self):
        await self.seed_review_draft()

        update = await ReflectReviewNode(
            artifact_store=self.store,
            model=Model(writing_review("revise", [writing_revision("synthesis", "review")])),
        )(self.state)

        self.assertEqual(update["stage"], "assembling_review")

    async def test_writing_revision_rerenders_only_its_target_section(self):
        framework = copy.deepcopy(FRAMEWORK)
        framework["sections"].append(
            {
                "section_id": "methods",
                "title": "Methods",
                "description": "How were the results obtained?",
                "discussion_questions": ["How were the results obtained?"],
                "relevant_paper_ids": ["1", "2"],
            }
        )
        await self.seed_framework(framework)
        self.state["claims_artifact_ref"] = await save(
            self.store, self.state, "claims", {"claims": [CLAIM], "unanswered": []}
        )
        self.state["claim_verification_artifact_ref"] = await save(
            self.store,
            self.state,
            "verification",
            {
                "claims": [
                    {
                        "claim_id": CLAIM["claim_id"],
                        "claim_hash": CLAIM["claim_hash"],
                        "status": "supported",
                        "reason": "Supported.",
                        "evidence": [
                            {
                                "evidence_id": "support_1",
                                "paper_id": "1",
                                "chunk_id": "chunk1",
                                "relation": "supports",
                                "text": "The result is conditional.",
                            },
                            {
                                "evidence_id": "support_2",
                                "paper_id": "2",
                                "chunk_id": "chunk2",
                                "relation": "supports",
                                "text": "The result is conditional.",
                            },
                        ],
                    }
                ]
            },
        )
        old_results = await save(
            self.store,
            self.state,
            "section_draft",
            {"section_id": "results", "summary": "Old results.", "input_hash": "old"},
        )
        old_methods = await save(
            self.store,
            self.state,
            "section_draft",
            {"section_id": "methods", "summary": "Old methods.", "input_hash": "old"},
        )
        self.state["section_draft_artifact_refs"] = {
            "results": old_results,
            "methods": old_methods,
        }
        self.state["revision_items"] = [writing_revision()]
        model = Model(
            {
                "arguments": [
                    {
                        "text": "The results depend on conditions [[REF_1]] [[REF_2]].",
                        "claim_ids": ["claim_001"],
                        "evidence_ids": ["support_1", "support_2"],
                    }
                ],
                "summary": "Updated results.",
            }
        )

        update = await RenderSectionsNode(artifact_store=self.store, model=model)(self.state)

        self.assertEqual(update["stage"], "assembling_review")
        self.assertNotEqual(update["section_draft_artifact_refs"]["results"], old_results)
        self.assertEqual(update["section_draft_artifact_refs"]["methods"], old_methods)
        self.assertEqual(update["revision_items"], [])
        self.assertEqual(len(model.inputs), 1)

    async def test_writing_reflection_rejects_research_stage_revisions(self):
        await self.seed_review_draft()
        invalid = writing_revision()
        invalid["target_type"] = "claim"

        update = await ReflectReviewNode(
            artifact_store=self.store,
            model=Model(writing_review("revise", [invalid])),
        )(self.state)

        self.assertEqual(update["stage"], "failed")
        self.assertEqual(update["error_code"], "REFLECTION_FAILED")

    def test_writing_review_contract_rejects_invalid_decision(self):
        with self.assertRaises(ValueError):
            WritingReviewResult.model_validate(writing_review("pass", [writing_revision()]))

    def test_claim_reflection_budget_default_and_bounds(self):
        self.assertEqual(TaskReviewDelegation(task_id=1, topic="topic").max_reflection_rounds, 5)
        for value in (1, 5):
            self.assertEqual(
                TaskReviewDelegation(
                    task_id=1,
                    topic="topic",
                    max_reflection_rounds=value,
                ).max_reflection_rounds,
                value,
            )
        for value in (0, 6):
            with self.assertRaises(ValueError):
                TaskReviewDelegation(
                    task_id=1,
                    topic="topic",
                    max_reflection_rounds=value,
                )

    def test_writing_revision_budget_default_and_bounds(self):
        self.assertEqual(TaskReviewDelegation(task_id=1, topic="topic").max_writing_revision_rounds, 1)
        for value in (0, 5):
            self.assertEqual(
                TaskReviewDelegation(
                    task_id=1,
                    topic="topic",
                    max_writing_revision_rounds=value,
                ).max_writing_revision_rounds,
                value,
            )
        with self.assertRaises(ValueError):
            TaskReviewDelegation(task_id=1, topic="topic", max_writing_revision_rounds=6)

    async def test_initialize_passes_writing_revision_budget_to_state(self):
        update = await initialize_review_node(
            {
                "run_id": "review-test",
                "active_tool_call": {
                    "args": {
                        "task_id": 1,
                        "topic": "topic",
                        "max_writing_revision_rounds": 2,
                    }
                },
            }
        )

        self.assertEqual(update["max_writing_revision_rounds"], 2)
        self.assertEqual(update["writing_revision_round"], 0)
        self.assertIsNone(update["writing_review_plan_ref"])

    def test_writing_reflection_route_cannot_return_to_research(self):
        for stage in ("generating_framework", "generating_claims", "retrieving_evidence"):
            self.assertEqual(_route_after_writing_review({"stage": stage}), "finalize")

    def test_claim_verification_route_cannot_return_to_framework_or_retrieval(self):
        for stage in ("generating_framework", "retrieving_evidence", "assembling_review"):
            self.assertEqual(_route_after_verification({"stage": stage}), "finalize")

    def test_reflection_allows_an_uncited_material_insufficiency_review(self):
        issues = ReflectReviewNode()._check_hard_constraints(
            review_draft={
                "body_markdown": "Current material is insufficient.",
                "abstract": "Current material is insufficient.",
                "conclusion": "Current material is insufficient.",
                "sections": [{"section_id": "results"}],
            },
            corpus_payload={"papers": []},
            framework=FRAMEWORK,
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

    async def test_workflow_skips_persistence_after_failed_handoff(self):
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
            "persist_review": "completed",
            "finalize_result": "completed",
        }

        def node(name):
            async def run(state):
                order.append(name)
                if name == "finalizing_handoff":
                    return {
                        "stage": "failed",
                        "error": "review was not approved by reflection",
                    }
                return {"stage": transitions[name]}

            return run

        graph = build_task_review_workflow(
            node_overrides={name: node(name) for name in [
                *transitions,
                "finalizing_handoff",
            ]}
        )
        result = await graph.ainvoke({})

        self.assertEqual(result["error"], "review was not approved by reflection")
        self.assertNotIn("persist_review", order)


class ReviewPromptSchemaTests(unittest.TestCase):
    schemas = (
        SearchCurrentPaperArgs,
        ProfileField,
        SubmittedArticleProfile,
        ExcludedPaper,
        FrameworkSection,
        ReviewFramework,
        CandidateClaim,
        ClaimsPlan,
        ClaimRevisionPlan,
        EvidenceQuote,
        ClaimRevision,
        Verdict,
        Argument,
        SectionDraft,
        ReviewSynthesis,
        WritingRevision,
        WritingReviewResult,
    )

    @staticmethod
    def json_examples(prompt):
        pattern = r"```json\s*(\{.*?\})\s*```"
        return [json.loads(value) for value in re.findall(pattern, prompt, re.DOTALL)]

    def test_model_visible_fields_have_descriptions(self):
        for schema_model in self.schemas:
            with self.subTest(schema=schema_model.__name__):
                schema = schema_model.model_json_schema()
                for model in (schema, *schema.get("$defs", {}).values()):
                    for name, field in model.get("properties", {}).items():
                        self.assertTrue(
                            field.get("description"),
                            f"{schema_model.__name__}.{name}",
                        )

    def test_schema_descriptions_and_prompt_examples(self):
        instruction = build_json_mode_instruction(SectionDraft)
        self.assertIn("章节正文中绑定 Claim 与证据引用的一个论证单元", instruction)
        self.assertIn("本节 arguments 的简短衔接性概述", instruction)

        search_parameters = ARTICLE_PROFILE_TOOLS[0]["function"]["parameters"]
        submit_parameters = ARTICLE_PROFILE_TOOLS[1]["function"]["parameters"]
        self.assertTrue(search_parameters["properties"]["query"].get("description"))
        self.assertTrue(submit_parameters["properties"]["core_problem"].get("description"))

        extract_examples = self.json_examples(ARTICLE_PROFILE_PROMPT)
        SearchCurrentPaperArgs.model_validate(extract_examples[0])
        SubmittedArticleProfile.model_validate(extract_examples[1])
        SubmittedArticleProfile.model_validate(
            self.json_examples(PROFILE_SUBMISSION_PROMPT)[0]
        )
        for prompt, model in (
            (FRAMEWORK_PROMPT, ReviewFramework),
            (CLAIM_GENERATION_PROMPT, ClaimsPlan),
            (CLAIM_REVISION_PROMPT, ClaimRevisionPlan),
            (CLAIM_VERIFICATION_PROMPT, Verdict),
            (SECTION_DRAFT_PROMPT, SectionDraft),
            (ASSEMBLE_REVIEW_PROMPT, ReviewSynthesis),
            (REPAIR_SYNTHESIS_PROMPT, ReviewSynthesis),
            (WRITING_REVIEW_PROMPT, WritingReviewResult),
        ):
            for example in self.json_examples(prompt):
                model.model_validate(example)


if __name__ == "__main__":
    unittest.main()
