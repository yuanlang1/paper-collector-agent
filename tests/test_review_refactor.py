import asyncio
import copy
import json
import re
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.documents import Document

from app.config import Settings, settings

from app.events.adapter import event_data, sse_name_for
from app.events.bus import EventBus
from app.events.context import EventContext, bind_event_context
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.contracts import (
    CandidateClaim,
    claim_hash,
    content_hash,
    save,
)
from app.llm.graph.workflows.review_generate.nodes.assemble_review import (
    ASSEMBLE_REVIEW_PROMPT,
    ReviewSynthesis,
    SynthesisFragment,
)
from app.llm.graph.workflows.review_generate.nodes.extract_studies import (
    ARTICLE_PROFILE_PROMPT,
    SubmittedArticleProfile,
)
from app.llm.tools.search_tools.search_paper_index import (
    SearchPaperIndexArgs,
)
from app.llm.graph.workflows.review_generate.nodes.generate_claim import (
    CLAIM_GENERATION_PROMPT,
    CLAIM_REVISION_PROMPT,
    ClaimRevisionPlan,
    ClaimsPlan,
    GenerateClaimsNode,
    claim_budgets,
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
from app.llm.graph.workflows.review_generate.nodes.retrieve_evidence import (
    RETRIEVAL_TOP_K,
    RetrieveEvidenceNode,
    select_evidence_snippets,
)
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
from app.llm.graph.workflows.task_indexing.workflow import (
    build_task_indexing_workflow,
)
from app.llm.graph.workflows.task_indexing.nodes.common import (
    total_progress_payload,
)
from app.llm.provider.structured_output import build_json_mode_instruction
from app.llm.subagents.task_review import (
    TaskReviewParams,
    task_review_agent,
)


def structured_response(messages, response):
    for prompt, schema in (
        (FRAMEWORK_PROMPT, ReviewFramework),
        (CLAIM_GENERATION_PROMPT, ClaimsPlan),
        (CLAIM_REVISION_PROMPT, ClaimRevisionPlan),
        (CLAIM_VERIFICATION_PROMPT, Verdict),
        (SECTION_DRAFT_PROMPT, SectionDraft),
        (WRITING_REVIEW_PROMPT, WritingReviewResult),
        (ASSEMBLE_REVIEW_PROMPT, ReviewSynthesis),
    ):
        if messages[0].content.startswith(prompt):
            return schema.model_validate(copy.deepcopy(response))
    raise AssertionError("unexpected structured model prompt")


class Model:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.inputs = []

    async def ainvoke(self, messages):
        self.inputs.append(json.loads(messages[1].content))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return structured_response(messages, response)


class CountingArtifactStore(LocalArtifactStore):
    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.reads = []

    async def read_json_uri(self, artifact_uri):
        self.reads.append(artifact_uri)
        return await super().read_json_uri(artifact_uri)


class ConcurrentModel:
    def __init__(self, responses, delays=None):
        self.responses = {claim_id: list(items) for claim_id, items in responses.items()}
        self.delays = delays or {}
        self.inputs = []
        self.active = 0
        self.peak = 0
        self.cancelled = []
        self.started = asyncio.Event()

    async def ainvoke(self, messages):
        payload = json.loads(messages[1].content)
        subject = (
            payload.get("claim")
            or payload.get("old_claim")
            or payload["section"]
        )
        claim_id = subject["claim_id"] if "claim_id" in subject else subject["section_id"]
        self.inputs.append(payload)
        self.started.set()
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delays.get(claim_id, 0))
            response = self.responses[claim_id].pop(0)
            if isinstance(response, Exception):
                raise response
            return structured_response(messages, response)
        except asyncio.CancelledError:
            self.cancelled.append(claim_id)
            raise
        finally:
            self.active -= 1


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


def verdict_revision(kind="claim"):
    return {
        "target_type": kind,
        "issue": "Missing comparison",
        "required_change": "Find conditional outcomes",
        "acceptance_criteria": "Comparable evidence",
    }


def verdict(status="supported"):
    return {
        "status": status,
        "evidence": [
            {
                "source_index": index,
                "assessment": "The result is conditional under the reported condition.",
                "relation": "supports",
            }
            for index in (1, 2)
        ],
        "reason": "Both original texts support conditionality",
        "revisions": [] if status == "supported" else [verdict_revision()],
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
        self.enterContext(patch.object(settings, "TASK_REVIEW_MAX_REFLECTION_ROUNDS", 3))
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = LocalArtifactStore(self.directory.name)
        self.state = {
            "run_id": "review-test",
            "child_run_id": "child-review-test",
            "task_id": 1,
            "topic": "Conditional outcomes",
            "language": "en",
            "review_type": "narrative",
            "paper_ids_snapshot": ["1", "2"],
            "reflection_round": 0,
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
        self.state["corpus_artifact_ref"] = await save(
            self.store,
            self.state,
            "corpus",
            {
                "papers": [
                    {"paper_id": 1, "title": "One"},
                    {"paper_id": 2, "title": "Two"},
                ]
            },
        )

    async def seed_framework(self, framework=FRAMEWORK):
        self.state["framework_artifact_ref"] = await save(
            self.store,
            self.state,
            "framework",
            {
                "framework": framework,
            },
        )

    async def seed_claim(self):
        await self.seed_framework()
        self.state["claims_artifact_ref"] = await save(
            self.store,
            self.state,
            "claims",
            {"claims": [CLAIM], "unanswered": []},
        )

    async def seed_claims(self, count):
        await self.seed_framework()
        claims = []
        for index in range(1, count + 1):
            claim = dict(
                CLAIM,
                claim_id=f"claim_{index:03d}",
                text=f"The result depends on reported condition {index}.",
            )
            claim["claim_hash"] = claim_hash(claim)
            claims.append(claim)
        self.state["claims_artifact_ref"] = await save(
            self.store,
            self.state,
            "claims",
            {"claims": claims, "unanswered": []},
        )
        return claims

    @staticmethod
    def framework_with_sections(count):
        return {
            **FRAMEWORK,
            "sections": [
                {
                    **FRAMEWORK["sections"][0],
                    "section_id": f"section_{index}",
                    "title": f"Section {index}",
                }
                for index in range(1, count + 1)
            ],
        }

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
                "abstract": "Abstract [@paper-1].",
                "sections": [
                    {
                        "section_id": "results",
                        "title": "Results",
                        "text": "The result depends on study conditions [@paper-1; @paper-2].",
                        "summary": "Conditional outcomes.",
                        "arguments": [],
                        "used_claim_ids": ["claim_001"],
                        "omitted_claim_ids": [],
                    }
                ],
                "body_markdown": "## Results\n\nThe result depends on study conditions [@paper-1; @paper-2].",
                "conclusion": "The outcome is conditional [@paper-1].",
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
        self.assertNotIn("framework_input_hash", framework)
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
        self.assertNotIn("changed_section_ids", claims)

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

    async def test_claim_generation_limits_section_concurrency_and_keeps_order(self):
        framework = self.framework_with_sections(5)
        await self.seed_framework(framework)
        model = ConcurrentModel(
            {
                section["section_id"]: [
                    {
                        "claims": [
                            dict(
                                CANDIDATE,
                                text=f"Section {index} has a distinct conditional result.",
                            )
                        ]
                    }
                ]
                for index, section in enumerate(framework["sections"], start=1)
            },
            {
                "section_1": 0.04,
                "section_2": 0.03,
                "section_3": 0.02,
                "section_4": 0.01,
            },
        )

        update = await GenerateClaimsNode(
            artifact_store=self.store, model=model, max_concurrency=4
        )(self.state)

        self.assertEqual(model.peak, 4)
        claims = await self.store.read_json_uri(update["claims_artifact_ref"])
        self.assertEqual(
            [claim["section_id"] for claim in claims["claims"]],
            [section["section_id"] for section in framework["sections"]],
        )
        self.assertEqual(
            [claim["claim_id"] for claim in claims["claims"]],
            [f"claim_{index:03d}" for index in range(1, 6)],
        )

    def test_claim_budgets_share_the_review_limit_by_framework_order(self):
        framework = self.framework_with_sections(7)

        self.assertEqual(
            claim_budgets(framework["sections"], []),
            {
                "section_1": 4,
                "section_2": 4,
                "section_3": 4,
                "section_4": 3,
                "section_5": 3,
                "section_6": 3,
                "section_7": 3,
            },
        )

    async def test_claim_generation_limits_model_output_to_the_section_budget(self):
        candidates = [
            dict(CANDIDATE, text=f"Conditional result {index} has a distinct scope.")
            for index in range(1, 6)
        ]
        await self.seed_framework()
        model = Model({"claims": candidates})

        update = await GenerateClaimsNode(artifact_store=self.store, model=model)(self.state)

        payload = await self.store.read_json_uri(update["claims_artifact_ref"])
        self.assertEqual(len(payload["claims"]), 4)
        self.assertIn("Claim budget limited section results to 4 claims.", update["warnings"])
        self.assertEqual(model.inputs[0]["claim_budget"], 4)

    async def test_add_claim_is_skipped_when_the_section_budget_is_full(self):
        await self.seed_claims(4)
        self.state["revision_items"] = [revision("add_claim", "results")]
        model = Model()

        update = await GenerateClaimsNode(artifact_store=self.store, model=model)(self.state)

        self.assertEqual(model.inputs, [])
        payload = await self.store.read_json_uri(update["claims_artifact_ref"])
        self.assertEqual(len(payload["claims"]), 4)
        self.assertEqual(payload["unanswered"][0]["reason"], "Claim budget exhausted for this section.")

    async def test_claim_generation_cancels_pending_sections_after_a_failure(self):
        framework = self.framework_with_sections(3)
        await self.seed_framework(framework)
        model = ConcurrentModel(
            {
                "section_1": [{"claims": [CANDIDATE]}],
                "section_2": [RuntimeError("connection lost")],
                "section_3": [{"claims": [CANDIDATE]}],
            },
            {"section_1": 0.01, "section_2": 0.04, "section_3": 0.2},
        )

        update = await GenerateClaimsNode(
            artifact_store=self.store, model=model, max_concurrency=3
        )(self.state)

        self.assertEqual(update["error_code"], "CLAIMS_FAILED")
        self.assertEqual(model.cancelled, ["section_3"])

    async def test_existing_framework_skips_generation_without_reading_artifacts(self):
        first = Model(FRAMEWORK)
        self.state.update(await GenerateFrameworkNode(artifact_store=self.store, model=first)(self.state))
        self.state["topic"] = "Changed topic"
        self.state["study_records_artifact_ref"] = "missing-studies"
        self.state["framework_artifact_ref"] = "existing-framework"
        cached = Model()
        update = await GenerateFrameworkNode(artifact_store=self.store, model=cached)(self.state)
        self.assertEqual(update["stage"], "generating_claims")
        self.assertEqual(cached.inputs, [])

    async def test_claim_revision_keeps_the_claim_id_and_reenters_verification(self):
        await self.seed_claim()
        await self.retrieve()
        self.state["revision_items"] = [revision("claim")]
        revision_model = Model(
            {"claims": [dict(CANDIDATE, text="Results depend on sample size.")]}
        )
        update = await GenerateClaimsNode(
            artifact_store=self.store, model=Model(), revision_model=revision_model
        )(self.state)
        payload = await self.store.read_json_uri(update["claims_artifact_ref"])
        self.assertEqual([claim["claim_id"] for claim in payload["claims"]], ["claim_001"])
        self.assertEqual(payload["claims"][0]["section_id"], "results")
        self.assertEqual(
            revision_model.inputs[0]["section"]["discussion_questions"],
            ["How do study conditions affect outcomes?"],
        )
        self.state.update(update)
        await self.retrieve()
        result = await VerifyClaimsNode(
            artifact_store=self.store, model=Model(verdict(), verdict()), max_concurrency=1
        )(self.state)
        self.assertEqual(result["stage"], "rendering_sections")

    async def test_claim_revisions_are_concurrent_and_read_the_ledger_once(self):
        claims = await self.seed_claims(5)
        await self.retrieve()
        self.state["revision_items"] = [
            revision("claim", claim["claim_id"]) for claim in claims
        ]
        store = CountingArtifactStore(self.directory.name)
        model = ConcurrentModel(
            {
                claim["claim_id"]: [
                    {
                        "claims": [
                            dict(
                                CANDIDATE,
                                text=f"Revised condition {index} is evidence-backed.",
                            )
                        ]
                    }
                ]
                for index, claim in enumerate(claims, start=1)
            },
            {
                "claim_001": 0.04,
                "claim_002": 0.03,
                "claim_003": 0.02,
                "claim_004": 0.01,
            },
        )

        update = await GenerateClaimsNode(
            artifact_store=store,
            model=Model(),
            revision_model=model,
            max_concurrency=4,
        )(self.state)

        self.assertEqual(model.peak, 4)
        self.assertEqual(
            store.reads.count(self.state["evidence_ledger_artifact_ref"]), 1
        )
        payload = await store.read_json_uri(update["claims_artifact_ref"])
        self.assertEqual(
            [claim["claim_id"] for claim in payload["claims"]],
            [claim["claim_id"] for claim in claims],
        )

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

    async def test_add_claim_continues_after_existing_ids_with_gaps(self):
        await self.seed_claim()
        self.state["claims_artifact_ref"] = await save(
            self.store, self.state, "claims",
            {"claims": [dict(CLAIM, claim_id="claim_003")], "unanswered": []},
        )
        self.state["revision_items"] = [revision("add_claim", "results")]
        update = await GenerateClaimsNode(
            artifact_store=self.store, model=Model({"claims": [CANDIDATE]}),
        )(self.state)
        payload = await self.store.read_json_uri(update["claims_artifact_ref"])
        self.assertEqual([claim["claim_id"] for claim in payload["claims"]], ["claim_003", "claim_004"])

    async def test_claim_generation_rejects_model_papers_outside_task(self):
        await self.seed_framework()
        update = await GenerateClaimsNode(
            artifact_store=self.store,
            model=Model({"claims": [dict(CANDIDATE, candidate_paper_ids=["99"])]}),
        )(self.state)
        self.assertEqual(update["error_code"], "CLAIMS_FAILED")
        self.assertIn("papers outside task", update["error"])

    async def test_supported_comparison_with_one_source_revises_the_claim(self):
        await self.seed_claim()
        await self.retrieve([document("1"), document("1", "chunk2")])
        self.state["reflection_round"] = settings.TASK_REVIEW_MAX_REFLECTION_ROUNDS - 2
        result = verdict()
        result["evidence"] = result["evidence"][:1]
        model = Model(result)
        update = await VerifyClaimsNode(
            artifact_store=self.store,
            model=model,
        )(self.state)
        self.assertEqual(update["stage"], "generating_claims")
        self.assertEqual(update["revision_items"][0]["target_type"], "claim")
        self.assertEqual(update["reflection_round"], settings.TASK_REVIEW_MAX_REFLECTION_ROUNDS - 1)
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(
            [item["paper_index"] for item in model.inputs[0]["original_evidence"]],
            [1, 1],
        )
        verification = await self.store.read_json_uri(
            update["claim_verification_artifact_ref"]
        )
        saved = verification["claims"][0]
        self.assertEqual(saved["status"], "insufficient")
        self.assertIn("1 supporting papers; 2 required", saved["reason"])
        self.assertNotIn("paper_index", saved["evidence"][0])

    async def test_one_support_and_one_opposition_do_not_meet_multiple_source_requirement(self):
        await self.seed_claim()
        await self.retrieve()
        result = verdict()
        result["evidence"][1]["relation"] = "opposes"
        model = Model(result)

        update = await VerifyClaimsNode(artifact_store=self.store, model=model)(self.state)

        self.assertEqual(update["stage"], "generating_claims")
        self.assertEqual(len(model.inputs), 1)
        verification = await self.store.read_json_uri(
            update["claim_verification_artifact_ref"]
        )
        self.assertEqual(verification["claims"][0]["status"], "insufficient")

    async def test_single_study_description_requires_only_one_supporting_paper(self):
        await self.seed_framework()
        claim = dict(
            CLAIM,
            claim_type="descriptive",
            evidence_requirement="fulltext",
            candidate_paper_ids=["1"],
        )
        claim["claim_hash"] = claim_hash(claim)
        self.state["claims_artifact_ref"] = await save(
            self.store, self.state, "claims", {"claims": [claim], "unanswered": []}
        )
        await self.retrieve([document("1"), document("1", "chunk2")])

        update = await self.verify(verdict())

        self.assertEqual(update["stage"], "rendering_sections")
        comparative = CandidateClaim.model_validate(
            dict(CANDIDATE, evidence_requirement="fulltext")
        )
        self.assertEqual(comparative.evidence_requirement, "multiple_fulltext")

    async def test_exhausted_evidence_revision_renders_material_insufficiency(self):
        await self.seed_claim()
        await self.retrieve()
        self.enterContext(patch.object(settings, "TASK_REVIEW_MAX_REFLECTION_ROUNDS", 1))
        self.state["reflection_round"] = 0

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
        result["revisions"] = [verdict_revision("add_claim")]
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

    async def test_verify_binds_revision_targets_to_current_context(self):
        await self.seed_claim()
        await self.retrieve()
        result = verdict("insufficient")
        result["revisions"] = [verdict_revision("claim")]
        update = await self.verify(result)
        self.assertEqual(update["stage"], "generating_claims")
        self.assertEqual(update["revision_items"][0]["target_id"], "claim_001")

    async def test_verify_restores_a_known_source_from_its_index(self):
        await self.seed_claim()
        await self.retrieve()
        result = verdict()
        result["evidence"][0]["assessment"] = "The result varies by condition."
        model = Model(result)

        update = await VerifyClaimsNode(artifact_store=self.store, model=model)(self.state)

        self.assertEqual(update["stage"], "rendering_sections")
        verification = await self.store.read_json_uri(
            update["claim_verification_artifact_ref"]
        )
        evidence = verification["claims"][0]["evidence"][0]
        self.assertEqual(evidence["text"], "The result is conditional.")
        self.assertEqual(evidence["char_start"], 0)
        self.assertEqual(evidence["char_end"], len(evidence["text"]))
        self.assertEqual(evidence["paper_id"], "1")
        self.assertNotIn("source_index", evidence)
        self.assertNotIn("paper_index", evidence)
        self.assertEqual(
            [item["paper_index"] for item in model.inputs[0]["original_evidence"]],
            [1, 2],
        )

    async def test_verify_corrects_an_invalid_source_index_once(self):
        await self.seed_claim()
        await self.retrieve()
        invalid = verdict()
        invalid["evidence"][0]["source_index"] = 3
        model = ConcurrentModel({"claim_001": [invalid, verdict()]})

        update = await VerifyClaimsNode(
            artifact_store=self.store, model=model
        )(self.state)

        self.assertEqual(update["stage"], "rendering_sections")
        self.assertEqual(len(model.inputs), 2)
        self.assertEqual(model.peak, 1)
        self.assertIn("validation_feedback", model.inputs[1])

    async def test_verify_limits_concurrency_and_keeps_claim_order(self):
        claims = await self.seed_claims(5)
        await self.retrieve()
        model = ConcurrentModel(
            {claim["claim_id"]: [verdict("insufficient")] for claim in claims},
            {
                "claim_001": 0.04,
                "claim_002": 0.03,
                "claim_003": 0.02,
                "claim_004": 0.01,
            },
        )

        update = await VerifyClaimsNode(
            artifact_store=self.store, model=model, max_concurrency=4
        )(self.state)

        self.assertEqual(model.peak, 4)
        self.assertEqual(update["stage"], "generating_claims")
        saved = await self.store.read_json_uri(update["claim_verification_artifact_ref"])
        self.assertEqual(
            [item["claim_id"] for item in saved["claims"]],
            [claim["claim_id"] for claim in claims],
        )
        self.assertEqual(
            [item["target_id"] for item in update["revision_items"]],
            [claim["claim_id"] for claim in claims],
        )

    async def test_insufficient_verdict_does_not_cancel_other_concurrent_claims(self):
        claims = await self.seed_claims(2)
        await self.retrieve()
        insufficient = verdict()
        insufficient["evidence"][1]["relation"] = "opposes"
        model = ConcurrentModel(
            {
                "claim_001": [insufficient],
                "claim_002": [verdict()],
            }
        )

        update = await VerifyClaimsNode(
            artifact_store=self.store, model=model, max_concurrency=2
        )(self.state)

        self.assertEqual(update["stage"], "generating_claims")
        self.assertEqual(model.cancelled, [])
        saved = await self.store.read_json_uri(update["claim_verification_artifact_ref"])
        self.assertEqual(
            [item["status"] for item in saved["claims"]], ["insufficient", "supported"]
        )

    async def test_verify_reuses_cached_verdicts_without_model_calls(self):
        claims = await self.seed_claims(2)
        await self.retrieve()
        initial = ConcurrentModel(
            {claim["claim_id"]: [verdict()] for claim in claims}
        )
        self.state.update(
            await VerifyClaimsNode(
                artifact_store=self.store, model=initial, max_concurrency=1
            )(self.state)
        )
        cached = ConcurrentModel({})

        update = await VerifyClaimsNode(
            artifact_store=self.store, model=cached, max_concurrency=4
        )(self.state)

        self.assertEqual(cached.inputs, [])
        saved = await self.store.read_json_uri(update["claim_verification_artifact_ref"])
        self.assertEqual(
            [item["claim_id"] for item in saved["claims"]],
            [claim["claim_id"] for claim in claims],
        )

    async def test_verify_preserves_completed_results_when_a_concurrent_call_fails(self):
        claims = await self.seed_claims(3)
        await self.retrieve()
        model = ConcurrentModel(
            {
                "claim_001": [verdict()],
                "claim_002": [RuntimeError("connection lost")],
                "claim_003": [verdict()],
            },
            {"claim_001": 0.01, "claim_002": 0.04, "claim_003": 0.2},
        )

        update = await VerifyClaimsNode(
            artifact_store=self.store, model=model, max_concurrency=3
        )(self.state)

        self.assertEqual(update["error_code"], "VERIFICATION_FAILED")
        saved = await self.store.read_json_uri(update["claim_verification_artifact_ref"])
        self.assertEqual([item["claim_id"] for item in saved["claims"]], ["claim_001"])
        self.assertEqual(model.cancelled, ["claim_003"])

    async def test_verify_keeps_completed_claims_when_a_later_claim_fails(self):
        await self.seed_framework()
        second = dict(CLAIM, claim_id="claim_002")
        second["claim_hash"] = claim_hash(second)
        self.state["claims_artifact_ref"] = await save(
            self.store,
            self.state,
            "claims",
            {"claims": [CLAIM, second], "unanswered": []},
        )
        await self.retrieve()
        model = Model(verdict(), RuntimeError("connection lost"))

        update = await VerifyClaimsNode(
            artifact_store=self.store, model=model, max_concurrency=1
        )(self.state)

        self.assertEqual(update["error_code"], "VERIFICATION_FAILED")
        saved = await self.store.read_json_uri(
            update["claim_verification_artifact_ref"]
        )
        self.assertEqual([item["claim_id"] for item in saved["claims"]], ["claim_001"])

        self.state.update(update)
        retry = Model(verdict())
        resumed = await VerifyClaimsNode(
            artifact_store=self.store, model=retry
        )(self.state)
        self.assertEqual(resumed["stage"], "rendering_sections")
        self.assertEqual(len(retry.inputs), 1)

    async def test_verify_rejects_non_claim_revision_types(self):
        await self.seed_claim()
        await self.retrieve()
        result = verdict("insufficient")
        result["revisions"] = [revision("framework", "review")]
        update = await VerifyClaimsNode(
            artifact_store=self.store, model=Model(result, result)
        )(self.state)
        self.assertEqual(update["error_code"], "VERIFICATION_PROTOCOL_FAILED")
        self.assertIn("claim_id=claim_001", update["error"])

    async def test_verify_rejects_an_added_claim_without_a_claim_revision(self):
        await self.seed_claim()
        await self.retrieve()
        result = verdict("insufficient")
        result["revisions"] = [verdict_revision("add_claim")]
        update = await VerifyClaimsNode(
            artifact_store=self.store, model=Model(result, result)
        )(self.state)
        self.assertEqual(update["error_code"], "VERIFICATION_PROTOCOL_FAILED")

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
        self.assertTrue(
            all(call.kwargs["top_k"] == RETRIEVAL_TOP_K for call in retriever.search.await_args_list)
        )

    async def test_retrieval_uses_only_two_queries_from_a_legacy_claim(self):
        await self.seed_claim()
        claim = dict(
            CLAIM,
            retrieval_queries=["first query", "second query", "legacy query"],
        )
        self.state["claims_artifact_ref"] = await save(
            self.store, self.state, "claims", {"claims": [claim], "unanswered": []}
        )
        retriever = AsyncMock()
        retriever.search.return_value = [document(), document("2", "chunk2")]

        update = await RetrieveEvidenceNode(
            artifact_store=self.store, content_retrieval=retriever
        )(self.state)

        self.assertEqual(retriever.search.await_count, 2)
        ledger = await self.store.read_json_uri(update["evidence_ledger_artifact_ref"])
        self.assertEqual(ledger["claims"][0]["queries"], ["first query", "second query"])

    async def test_retrieval_refreshes_a_legacy_cache_without_selection_settings(self):
        await self.seed_claim()
        self.state["evidence_ledger_artifact_ref"] = await save(
            self.store,
            self.state,
            "evidence_ledger",
            {
                "claims": [
                    {
                        "claim_id": CLAIM["claim_id"],
                        "claim_hash": CLAIM["claim_hash"],
                        "retrieval_plan_hash": content_hash(
                            {
                                "candidate_paper_ids": CLAIM["candidate_paper_ids"],
                                "retrieval_queries": CLAIM["retrieval_queries"],
                            }
                        ),
                        "queries": CLAIM["retrieval_queries"],
                        "chunk_snippets": [],
                    }
                ]
            },
        )
        retriever = AsyncMock()
        retriever.search.return_value = [document(), document("2", "chunk2")]

        await RetrieveEvidenceNode(artifact_store=self.store, content_retrieval=retriever)(
            self.state
        )

        retriever.search.assert_awaited_once()

    def test_evidence_selection_is_deduplicated_diverse_and_stable(self):
        selected = select_evidence_snippets(
            [
                [
                    document("1", "a"),
                    document("1", "b"),
                    document("2", "a"),
                    document("2", "b"),
                ],
                [
                    document("1", "a"),
                    document("3", "a"),
                    document("3", "b"),
                    document("4", "a"),
                ],
            ]
        )

        keys = [(item["paper_id"], item["chunk_id"]) for item in selected]
        self.assertEqual(keys[:2], [("1", "a"), ("3", "a")])
        self.assertEqual(len(keys), 6)
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(keys.count((paper_id, chunk_id)) == 1 for paper_id, chunk_id in keys))
        self.assertTrue(
            all(sum(item["paper_id"] == paper_id for item in selected) <= 2 for paper_id in {item["paper_id"] for item in selected})
        )

    async def test_retrieval_replaces_old_snippets_when_the_claim_changes(self):
        await self.seed_claim()
        await self.retrieve([document("1", "old")])
        changed = dict(CLAIM, text="The result depends on intervention duration.")
        changed["claim_hash"] = claim_hash(changed)
        self.state["claims_artifact_ref"] = await save(
            self.store, self.state, "claims", {"claims": [changed], "unanswered": []}
        )
        retriever = AsyncMock()
        retriever.search.return_value = [document("2", "new")]

        update = await RetrieveEvidenceNode(
            artifact_store=self.store, content_retrieval=retriever
        )(self.state)

        ledger = await self.store.read_json_uri(update["evidence_ledger_artifact_ref"])
        self.assertEqual(
            [(item["paper_id"], item["chunk_id"]) for item in ledger["claims"][0]["chunk_snippets"]],
            [("2", "new")],
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
            "paper_title": "One",
            "section_path": "Results",
            "text": "Shared result.",
            "assessment": "The result is shared across the two claims.",
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
                        "text": "Shared result.",
                        "claim_ids": ["claim_a"],
                        "evidence_indices": [1],
                    },
                    {
                        "text": "Shared result.",
                        "claim_ids": ["claim_b"],
                        "evidence_indices": [1],
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
                "paper_title": f"Paper {paper_id}",
                "section_path": "Results",
                "relation": "supports",
                "text": "e" * 25_000,
                "assessment": "The result supports the claim.",
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
                        "text": "Conditional outcome.",
                        "claim_ids": ["claim_001"],
                        "evidence_indices": [2, 1],
                    }
                ],
                "summary": "Conditional outcome.",
            }
        )

        update = await RenderSectionsNode(artifact_store=self.store, model=model)(self.state)

        self.assertEqual(update["stage"], "assembling_review")
        self.assertEqual(len(model.inputs), 1)
        evidence_input = model.inputs[0]["claims_with_evidence"][0]["evidence"]
        self.assertEqual([item["source_index"] for item in evidence_input], [1, 2])
        self.assertEqual(
            [item["paper_title"] for item in evidence_input],
            ["Paper 1", "Paper 2"],
        )
        draft = await self.store.read_json_uri(
            update["section_draft_artifact_refs"]["results"]
        )
        self.assertEqual(draft["arguments"][0]["evidence_ids"], ["evidence_2", "evidence_1"])
        self.assertIn("[@paper-2; @paper-1]", draft["text"])

    async def test_section_rendering_limits_concurrency_and_keeps_section_order(self):
        framework = self.framework_with_sections(5)
        await self.seed_framework(framework)
        claims = []
        verdicts = []
        for index, section in enumerate(framework["sections"], start=1):
            claim = dict(
                CLAIM,
                claim_id=f"claim_{index:03d}",
                section_id=section["section_id"],
                text=f"Section {index} has an evidence-backed outcome.",
            )
            claim["claim_hash"] = claim_hash(claim)
            claims.append(claim)
            verdicts.append(
                {
                    "claim_id": claim["claim_id"],
                    "claim_hash": claim["claim_hash"],
                    "status": "supported",
                    "reason": "Supported.",
                    "evidence": [
                        {
                            "evidence_id": f"evidence_{index}_1",
                            "paper_id": "1",
                            "chunk_id": f"chunk_{index}_1",
                            "paper_title": "One",
                            "section_path": "Results",
                            "relation": "supports",
                            "text": "The result is conditional.",
                            "assessment": "The result supports the claim.",
                        },
                        {
                            "evidence_id": f"evidence_{index}_2",
                            "paper_id": "2",
                            "chunk_id": f"chunk_{index}_2",
                            "paper_title": "Two",
                            "section_path": "Results",
                            "relation": "supports",
                            "text": "The result is conditional.",
                            "assessment": "The result supports the claim.",
                        },
                    ],
                }
            )
        self.state["claims_artifact_ref"] = await save(
            self.store, self.state, "claims", {"claims": claims, "unanswered": []}
        )
        self.state["claim_verification_artifact_ref"] = await save(
            self.store, self.state, "verification", {"claims": verdicts}
        )
        model = ConcurrentModel(
            {
                section["section_id"]: [
                    {
                        "arguments": [
                            {
                                "text": f"Section {index} outcome.",
                                "claim_ids": [f"claim_{index:03d}"],
                                "evidence_indices": [1, 2],
                            }
                        ],
                        "summary": f"Section {index} summary.",
                    }
                ]
                for index, section in enumerate(framework["sections"], start=1)
            },
            {
                "section_1": 0.04,
                "section_2": 0.03,
                "section_3": 0.02,
                "section_4": 0.01,
            },
        )

        update = await RenderSectionsNode(
            artifact_store=self.store, model=model, max_concurrency=4
        )(self.state)

        self.assertEqual(model.peak, 4)
        self.assertEqual(
            list(update["section_draft_artifact_refs"]),
            [section["section_id"] for section in framework["sections"]],
        )
        self.assertTrue(
            all(
                payload["section_outline"]
                == model.inputs[0]["section_outline"]
                and "previous_section_summary" not in payload
                for payload in model.inputs
            )
        )

    async def test_section_outline_invalidates_a_cached_draft(self):
        await self.seed_claim()
        await self.retrieve()
        await self.verify(verdict())
        result = {
            "arguments": [
                {
                    "text": "Conditional outcome.",
                    "claim_ids": ["claim_001"],
                    "evidence_indices": [1, 2],
                }
            ],
            "summary": "Conditional outcome.",
        }
        first = await RenderSectionsNode(
            artifact_store=self.store, model=Model(result)
        )(self.state)
        self.state.update(first)

        cached = Model()
        second = await RenderSectionsNode(artifact_store=self.store, model=cached)(
            self.state
        )
        self.assertEqual(cached.inputs, [])

        framework = copy.deepcopy(FRAMEWORK)
        framework["sections"][0]["description"] = "A revised scope."
        await self.seed_framework(framework)
        refreshed = Model(result)
        update = await RenderSectionsNode(
            artifact_store=self.store, model=refreshed
        )(self.state)

        self.assertEqual(second["stage"], "assembling_review")
        self.assertEqual(len(refreshed.inputs), 1)
        self.assertEqual(
            refreshed.inputs[0]["section_outline"][0]["description"], "A revised scope."
        )

    async def test_section_rendering_keeps_completed_drafts_when_a_task_fails(self):
        framework = self.framework_with_sections(3)
        await self.seed_framework(framework)
        claims = []
        verdicts = []
        for index, section in enumerate(framework["sections"], start=1):
            claim = dict(
                CLAIM,
                claim_id=f"claim_{index:03d}",
                section_id=section["section_id"],
                text=f"Section {index} has an evidence-backed outcome.",
            )
            claim["claim_hash"] = claim_hash(claim)
            claims.append(claim)
            verdicts.append(
                {
                    "claim_id": claim["claim_id"],
                    "claim_hash": claim["claim_hash"],
                    "status": "supported",
                    "reason": "Supported.",
                    "evidence": [
                        {
                            "evidence_id": f"evidence_{index}_1",
                            "paper_id": "1",
                            "chunk_id": f"chunk_{index}_1",
                            "paper_title": "One",
                            "section_path": "Results",
                            "relation": "supports",
                            "text": "The result is conditional.",
                            "assessment": "The result supports the claim.",
                        },
                        {
                            "evidence_id": f"evidence_{index}_2",
                            "paper_id": "2",
                            "chunk_id": f"chunk_{index}_2",
                            "paper_title": "Two",
                            "section_path": "Results",
                            "relation": "supports",
                            "text": "The result is conditional.",
                            "assessment": "The result supports the claim.",
                        },
                    ],
                }
            )
        self.state["claims_artifact_ref"] = await save(
            self.store, self.state, "claims", {"claims": claims, "unanswered": []}
        )
        self.state["claim_verification_artifact_ref"] = await save(
            self.store, self.state, "verification", {"claims": verdicts}
        )
        response = {
            "arguments": [
                {
                    "text": "Section one outcome.",
                    "claim_ids": ["claim_001"],
                    "evidence_indices": [1, 2],
                }
            ],
            "summary": "Section one summary.",
        }
        model = ConcurrentModel(
            {
                "section_1": [response],
                "section_2": [RuntimeError("connection lost")],
                "section_3": [response],
            },
            {"section_1": 0.01, "section_2": 0.04, "section_3": 0.2},
        )

        update = await RenderSectionsNode(
            artifact_store=self.store, model=model, max_concurrency=3
        )(self.state)

        self.assertEqual(update["error_code"], "RENDER_FAILED")
        self.assertIn("section_1", update["section_draft_artifact_refs"])
        self.assertEqual(model.cancelled, ["section_3"])

    async def test_section_rendering_propagates_outer_cancellation(self):
        await self.seed_claim()
        await self.retrieve()
        await self.verify(verdict())
        model = ConcurrentModel(
            {
                "results": [
                    {
                        "arguments": [
                            {
                                "text": "Conditional outcome.",
                                "claim_ids": ["claim_001"],
                                "evidence_indices": [1, 2],
                            }
                        ],
                        "summary": "Conditional outcome.",
                    }
                ]
            },
            {"results": 0.2},
        )
        task = asyncio.create_task(
            RenderSectionsNode(artifact_store=self.store, model=model)(self.state)
        )
        await model.started.wait()
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(model.cancelled, ["results"])

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
        self.enterContext(patch.object(settings, "TASK_REVIEW_MAX_WRITING_REVISION_ROUNDS", 0))

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
        self.enterContext(patch.object(settings, "TASK_REVIEW_MAX_WRITING_REVISION_ROUNDS", 2))
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
                                "paper_title": "One",
                                "section_path": "Results",
                                "relation": "supports",
                                "text": "The result is conditional.",
                                "assessment": "The result supports the claim.",
                            },
                            {
                                "evidence_id": "support_2",
                                "paper_id": "2",
                                "chunk_id": "chunk2",
                                "paper_title": "Two",
                                "section_path": "Results",
                                "relation": "supports",
                                "text": "The result is conditional.",
                                "assessment": "The result supports the claim.",
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
                        "text": "The results depend on conditions.",
                        "claim_ids": ["claim_001"],
                        "evidence_indices": [1, 2],
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
        name = "TASK_REVIEW_MAX_REFLECTION_ROUNDS"
        self.assertEqual(Settings.model_fields[name].default, 5)
        for value in (1, 5):
            self.assertEqual(getattr(Settings.model_validate(settings.model_dump() | {name: value}), name), value)
        for value in (0, 6):
            with self.assertRaises(ValueError):
                Settings.model_validate(settings.model_dump() | {name: value})

    def test_review_subagent_does_not_require_confirmation(self):
        self.assertFalse(task_review_agent.metadata["requires_confirmation"])

    def test_writing_revision_budget_default_and_bounds(self):
        name = "TASK_REVIEW_MAX_WRITING_REVISION_ROUNDS"
        self.assertEqual(Settings.model_fields[name].default, 1)
        for value in (0, 5):
            self.assertEqual(getattr(Settings.model_validate(settings.model_dump() | {name: value}), name), value)
        for value in (-1, 6):
            with self.assertRaises(ValueError):
                Settings.model_validate(settings.model_dump() | {name: value})

    async def test_initialize_does_not_copy_writing_revision_budget_to_state(self):
        update = await initialize_review_node(
            {
                "run_id": "review-test",
                "child_run_id": "child-review-test",
                "task_id": 1,
                "topic": "topic",
                "max_writing_revision_rounds": 2,
            }
        )

        self.assertNotIn("max_writing_revision_rounds", update)
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
        with bind_event_context(EventContext(bus=EventBus(), run_id="review-test")):
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
        with bind_event_context(EventContext(bus=EventBus(), run_id="review-test")):
            result = await graph.ainvoke({})

        self.assertEqual(result["error"], "review was not approved by reflection")
        self.assertNotIn("persist_review", order)

    async def test_review_workflow_uses_progress_events_with_stable_iterations(self):
        stages = {
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
            async def run(_state):
                return {"stage": stages[name]}

            return run

        events = []

        async def collect(event):
            events.append(event)

        bus = EventBus()
        bus.subscribe(collect)
        graph = build_task_review_workflow(
            chat=Mock(),
            node_overrides={name: node(name) for name in stages},
        )
        with bind_event_context(EventContext(
            bus=bus,
            run_id="run-review",
            scope={
                "action_id": "review-1",
                "delegation_id": "review-1",
                "subagent": "task_review_agent",
                "workflow": "task_review",
            },
        )):
            await graph.ainvoke({
                "task_id": 8,
                "paper_ids_snapshot": ["1", "2"],
                "reflection_round": 2,
                "writing_revision_round": 1,
            })

        serialized = [
            {"name": sse_name_for(event), "data": event_data(event)}
            for event in events
        ]
        self.assertEqual(len(serialized), 26)
        self.assertTrue(all(item["name"] == "subagent_progress" for item in serialized))
        self.assertTrue(all(
            item["data"]["delegation_id"] == "review-1"
            for item in serialized
        ))
        for node_name, expected_iteration in (
            ("generate_claims", 3),
            ("render_sections", 2),
        ):
            node_events = [
                item["data"]
                for item in serialized
                if item["data"]["node"] == node_name
            ]
            self.assertEqual(
                [event["iteration"] for event in node_events],
                [expected_iteration, expected_iteration],
            )
            self.assertEqual(
                [event["data"]["phase_state"] for event in node_events],
                ["started", "completed"],
            )

    async def test_indexing_workflow_uses_progress_events_with_updated_counts(self):
        transitions = {
            "initialize": {"stage": "checking_status"},
            "check_rag_status": {"stage": "indexing"},
            "run_or_wait": {
                "stage": "verifying",
                "overall_summary": {"total": 2, "ready": 2},
                "committed_summary": {"total": 2, "ready": 2},
            },
            "verify_completion": {"stage": "completed"},
            "finalize_result": {"stage": "completed"},
        }

        def node(name):
            async def run(_state):
                return transitions[name]

            return run

        events = []

        async def collect(event):
            events.append(event)

        bus = EventBus()
        bus.subscribe(collect)
        graph = build_task_indexing_workflow(
            node_overrides={name: node(name) for name in transitions},
        )
        with bind_event_context(EventContext(
            bus=bus,
            run_id="run-index",
            scope={
                "action_id": "index-1",
                "delegation_id": "index-1",
                "subagent": "task_indexing_agent",
                "workflow": "task_indexing",
            },
        )):
            await graph.ainvoke({"task_id": 7})

        serialized = [
            {"name": sse_name_for(event), "data": event_data(event)}
            for event in events
        ]
        self.assertEqual(len(serialized), 10)
        self.assertTrue(all(item["name"] == "subagent_progress" for item in serialized))
        self.assertTrue(all(
            item["data"]["delegation_id"] == "index-1"
            for item in serialized
        ))
        self.assertTrue(all(
            "message" not in item["data"]
            for item in serialized
        ))
        self.assertTrue(all(
            "iteration" not in item["data"]
            for item in serialized
        ))
        run_or_wait_done = next(
            item["data"]
            for item in serialized
            if item["data"]["node"] == "run_or_wait"
            and item["data"]["data"]["phase_state"] == "completed"
        )
        self.assertEqual(
            run_or_wait_done["data"]["counts"],
            {"total": 2, "ready": 2},
        )

    async def test_indexing_live_progress_uses_only_progress_percent(self):
        payload = total_progress_payload(
            task_id=7,
            summary={"total": 2, "ready": 1, "skipped": 0, "failed": 0},
            progress_percent=50,
        )

        self.assertEqual(payload["progress_percent"], 50)
        self.assertNotIn("progress", payload)
        self.assertEqual(payload["data"]["phase_state"], "running")


class ReviewPromptSchemaTests(unittest.TestCase):
    schemas = (
        SearchPaperIndexArgs,
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
        SynthesisFragment,
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

        search_parameters = SearchPaperIndexArgs.model_json_schema()
        self.assertTrue(search_parameters["properties"]["querys"].get("description"))
        self.assertTrue(search_parameters["properties"]["paper_id"].get("description"))
        self.assertNotIn("quote 必须逐字", CLAIM_VERIFICATION_PROMPT)
        self.assertIn("概括", EvidenceQuote.model_fields["assessment"].description)
        self.assertIn("paper_index", CLAIM_VERIFICATION_PROMPT)
        self.assertIn("至少需要两篇 supports 的不同论文", CLAIM_VERIFICATION_PROMPT)
        self.assertIn("跨论文比较必须使用", CLAIM_GENERATION_PROMPT)
        self.assertIn("不按论文数量或讨论问题数量机械增加论点", CLAIM_GENERATION_PROMPT)
        self.assertIn("claim_budget", CLAIM_GENERATION_PROMPT)
        self.assertIn("默认只用一条", CLAIM_GENERATION_PROMPT)
        self.assertIn("不能只降低 evidence_requirement", CLAIM_REVISION_PROMPT)
        self.assertIn("只能返回一条", CLAIM_REVISION_PROMPT)
        self.assertIn("限定条件、反例、补充说明", CLAIM_VERIFICATION_PROMPT)
        self.assertIn("经过相关性筛选", CLAIM_VERIFICATION_PROMPT)
        self.assertIn("section_outline 是固定的综述目录", SECTION_DRAFT_PROMPT)
        self.assertEqual(
            CandidateClaim.model_json_schema()["properties"]["retrieval_queries"]["maxItems"],
            2,
        )
        self.assertEqual(
            ClaimRevisionPlan.model_json_schema()["properties"]["claims"]["maxItems"],
            1,
        )

        extract_examples = self.json_examples(ARTICLE_PROFILE_PROMPT)
        SearchPaperIndexArgs.model_validate(extract_examples[0])
        SubmittedArticleProfile.model_validate(extract_examples[1])
        for prompt, model in (
            (FRAMEWORK_PROMPT, ReviewFramework),
            (CLAIM_GENERATION_PROMPT, ClaimsPlan),
            (CLAIM_REVISION_PROMPT, ClaimRevisionPlan),
            (CLAIM_VERIFICATION_PROMPT, Verdict),
            (SECTION_DRAFT_PROMPT, SectionDraft),
            (ASSEMBLE_REVIEW_PROMPT, ReviewSynthesis),
            (WRITING_REVIEW_PROMPT, WritingReviewResult),
        ):
            for example in self.json_examples(prompt):
                model.model_validate(example)


if __name__ == "__main__":
    unittest.main()
