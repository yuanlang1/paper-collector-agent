import asyncio
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
    verified_pack,
)
from app.llm.graph.workflows.review_generate.nodes.extract_studies import ExtractStudiesNode
from app.llm.graph.workflows.review_generate.nodes.resolve_review_focus import (
    ResolveReviewFocusNode,
)
from app.llm.graph.workflows.review_generate.nodes.generate_claim import GenerateClaimsNode
from app.llm.graph.workflows.review_generate.nodes.retrieve_evidence import RetrieveEvidenceNode
from app.llm.graph.workflows.review_generate.nodes.verify_claims import VerifyClaimsNode
from app.llm.graph.workflows.review_generate.nodes.generate_framework import GenerateFrameworkNode
from app.llm.graph.workflows.review_generate.nodes.render_section import RenderSectionsNode
from app.llm.graph.workflows.review_generate.nodes.assemble_review import AssembleReviewNode
from app.llm.graph.workflows.review_generate.nodes.reflect_review import ReflectReviewNode
from app.llm.graph.workflows.review_generate.nodes.finalize import finalize_task_review_node
from app.llm.graph.workflows.review_generate.workflow import build_task_review_workflow
from app.llm.graph.workflows.review_generate.revisions import schedule_revision
from app.rag.index_construction.paper_reading_index_construction import (
    ArticleProfile,
    PaperReadingIndexConstructionModule,
)


class Model:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.inputs = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.inputs.append(json.loads(messages[1].content))
        return copy.deepcopy(self.responses.pop(0))


class ReadingIndex:
    def __init__(self):
        self.records = {}
        self.attempts = 0
        self.failures = 0

    async def lookup(self, paper_id):
        return self.records.get(paper_id)

    async def save(self, payload):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise ConnectionError("storage unavailable")
        self.records[payload["paper_id"]] = payload


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


PROFILE = ArticleProfile(
    core_problem="Problem", methods="Method", main_discussion="Conditional result"
)
FOCUS = dict(
    title="Conditional outcomes",
    scope="A fixed corpus synthesis",
    adjustment_reason="Remove presumed superiority",
    corpus_scope_mismatch=False,
    research_questions=[
        dict(
            question="Under what conditions?",
            core=True,
            relevant_paper_ids=["1", "2"],
            question_id="rq_001",
        )
    ],
    unanswerable_questions=[],
)
CANDIDATE = dict(
    text="The results depend on study conditions.",
    candidate_paper_ids=["1", "2"],
    retrieval_queries=["study conditions"],
    claim_type="comparative",
    evidence_requirement="multiple_fulltext",
    withdrawn_reason="",
)
CLAIM = dict(CANDIDATE, claim_id="claim_001", question_id="rq_001")
CLAIM["claim_hash"] = claim_hash(CLAIM)


def revision(kind="retrieval", target="claim_001"):
    return dict(
        target_type=kind,
        target_id=target,
        issue="Missing comparison",
        required_change="Find conditional outcomes",
        acceptance_criteria="Comparable evidence",
        queries=["conditional outcomes"],
    )


def verdict(status="supported"):
    return dict(
        status=status,
        evidence=[
            dict(
                paper_id=pid,
                chunk_id=f"chunk{pid}",
                quote="The result is conditional.",
                relation="supports",
            )
            for pid in ("1", "2")
        ],
        reason="Both original texts support conditionality",
        revisions=[] if status == "supported" else [revision()],
    )


class RefactorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = LocalArtifactStore(self.directory.name)
        self.state = dict(
            run_id="refactor-test",
            task_id=1,
            topic="Universal superiority",
            language="en",
            review_type="narrative",
            citation_style="ieee",
            paper_ids_snapshot=["1", "2"],
            max_reflection_rounds=3,
            reflection_round=0,
            review_focus=copy.deepcopy(FOCUS),
            revision_items=[],
            warnings=[],
        )

    async def seed(self):
        self.state["claims_artifact_ref"] = await save(
            self.store, self.state, "claims", {"claims": [CLAIM], "unanswered": []}
        )
        self.state["study_records_artifact_ref"] = await save(
            self.store,
            self.state,
            "studies",
            {"studies": [dict(paper_id=pid, **PROFILE.model_dump()) for pid in ("1", "2")]},
        )
        self.state["corpus_artifact_ref"] = await save(
            self.store,
            self.state,
            "corpus",
            {
                "papers": [
                    dict(paper_id=pid, title=pid, authors=[], published_date=None, doi="")
                    for pid in ("1", "2")
                ]
            },
        )

    def reading_node(self, index):
        reader = AsyncMock()
        reader.read.return_value = [document()]
        node = ExtractStudiesNode(
            artifact_store=self.store,
            model=Model(),
            corpus_reader=reader,
            content_retrieval=AsyncMock(),
            reading_index=index,
        )
        node._extract_profile = AsyncMock(return_value=PROFILE)
        return node

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

    async def test_cache_hit_skips_reading_after_source_changes(self):
        index = ReadingIndex()
        node = self.reading_node(index)
        await node._read_cached(paper={"paper_id": 1})
        payload = await node._read_cached(paper={"paper_id": 1})
        self.assertEqual(node._extract_profile.await_count, 1)
        self.assertEqual(set(payload), {"paper_id", "profile"})
        self.assertEqual(set(payload["profile"]), {"core_problem", "methods", "main_discussion"})
        node.corpus_reader.read.return_value = [document(text="Changed original")]
        await node._read_cached(paper={"paper_id": 1})
        self.assertEqual(node._extract_profile.await_count, 1)
        self.assertEqual(node.corpus_reader.read.await_count, 1)

        index.records.clear()
        await node._read_cached(paper={"paper_id": 1})
        self.assertEqual(node._extract_profile.await_count, 2)

    async def test_save_failure_retries_only_save_and_preserves_payload(self):
        index = ReadingIndex()
        index.failures = 2
        node = self.reading_node(index)
        await node._read_cached(paper={"paper_id": 1})
        self.assertEqual(index.attempts, 3)
        self.assertEqual(node._extract_profile.await_count, 1)

    async def test_same_key_reading_is_coalesced(self):
        index = ReadingIndex()
        node = self.reading_node(index)
        await asyncio.gather(*(node._read_cached(paper={"paper_id": 1}) for _ in range(5)))
        self.assertEqual(node._extract_profile.await_count, 1)
        self.assertEqual(index.attempts, 1)

    async def test_lookup_error_does_not_call_model(self):
        index = ReadingIndex()
        index.lookup = AsyncMock(side_effect=ConnectionError("offline"))
        node = self.reading_node(index)
        with self.assertRaises(ConnectionError):
            await node._read_cached(paper={"paper_id": 1})
        node._extract_profile.assert_not_awaited()

    async def test_lookup_distinguishes_absent_corrupt_and_connection_failure(self):
        client = AsyncMock()
        index = PaperReadingIndexConstructionModule(
            client=client, dense_embeddings=object(), sparse_embeddings=object()
        )
        client.collection_exists.return_value = False
        self.assertIsNone(await index.lookup("1"))
        client.collection_exists.return_value = True
        client.retrieve.return_value = [type("Record", (), {"payload": {"profile": {}}})()]
        self.assertIsNone(await index.lookup("1"))
        client.retrieve.side_effect = ConnectionError("offline")
        with self.assertRaises(ConnectionError):
            await index.lookup("1")

    async def test_reading_index_roundtrip_uses_existing_dense_sparse_base(self):
        from qdrant_client import AsyncQdrantClient, models

        client = AsyncQdrantClient(":memory:")
        embeddings = AsyncMock()
        embeddings.aembed_documents.return_value = [[1.0, 0.0]]
        index = PaperReadingIndexConstructionModule(
            client=client, dense_embeddings=embeddings, sparse_embeddings=object()
        )
        index._embed_sparse_documents = lambda texts: [
            models.SparseVector(indices=[1], values=[1.0])
        ]
        try:
            self.assertIsNone(await index.lookup("1"))
            payload = index.payload("1", PROFILE)
            await index.save(payload)
            self.assertEqual(await index.lookup("1"), payload)
            await index.save(payload)
            count = await client.count(collection_name=index.collection_name)
            self.assertEqual(count.count, 1)
        finally:
            await client.close()

    async def test_failed_save_does_not_reread_within_the_attempt(self):
        index = ReadingIndex()
        index.failures = 10
        node = self.reading_node(index)
        with self.assertRaises(ConnectionError):
            await node._read_cached(paper={"paper_id": 1})
        self.assertEqual(index.attempts, 3)
        self.assertEqual(node._extract_profile.await_count, 1)

    async def test_partial_cache_hit_and_other_failure_preserves_success(self):
        await self.seed()
        self.state.pop("study_records_artifact_ref")
        index = ReadingIndex()
        node = self.reading_node(index)
        await node._read_cached(paper={"paper_id": 1})
        node._extract_profile.side_effect = ValueError("unusable")
        update = await node({**self.state, "stage": "extracting_studies"})
        self.assertEqual(update["stage"], "failed")
        self.assertEqual(len(index.records), 1)
        report = await self.store.read_json_uri(update["study_extraction_report_artifact_ref"])
        self.assertEqual(report["extracted_paper_ids"], ["1"])
        self.assertEqual(report["failures"][0]["paper_id"], "2")

    async def test_full_cache_hit_has_no_reading_calls(self):
        await self.seed()
        self.state.pop("study_records_artifact_ref")
        node = self.reading_node(ReadingIndex())
        for pid in ("1", "2"):
            await node._read_cached(paper={"paper_id": pid})
        node._extract_profile.reset_mock()
        update = await node({**self.state, "stage": "extracting_studies"})
        self.assertEqual(update["stage"], "resolving_review_focus")
        node._extract_profile.assert_not_awaited()

    def test_claim_invariants(self):
        for claims in (
            [CLAIM, CLAIM],
            [dict(CLAIM, candidate_paper_ids=["99"])],
            [dict(CLAIM, question_id="missing")],
        ):
            with self.assertRaises(ValueError):
                validate_claims(claims, FOCUS, ["1", "2"])

    async def test_focus_corrects_title_and_preserves_topic(self):
        await self.seed()
        update = await ResolveReviewFocusNode(artifact_store=self.store, model=Model(FOCUS))(
            self.state
        )
        self.assertEqual(update["review_focus"]["title"], FOCUS["title"])
        self.assertEqual(self.state["topic"], "Universal superiority")
        update = await ResolveReviewFocusNode(
            artifact_store=self.store, model=Model(dict(FOCUS, corpus_scope_mismatch=True))
        )(self.state)
        self.assertEqual(update["error_code"], "CORPUS_SCOPE_MISMATCH")

    async def test_dual_retrieval_deduplicates_and_preserves_second_paper(self):
        await self.seed()
        self.state["claims_artifact_ref"] = await save(
            self.store,
            self.state,
            "claims",
            {"claims": [dict(CLAIM, candidate_paper_ids=["1"])], "unanswered": []},
        )
        retriever = await self.retrieve(
            [document(chunk_id=f"a{i}") for i in range(3)] + [document("2", "chunk2")]
        )
        self.assertEqual(retriever.search.await_count, 2)
        ledger = await self.store.read_json_uri(self.state["evidence_ledger_artifact_ref"])
        self.assertEqual(len(ledger["claims"][0]["chunk_snippets"]), 4)
        self.assertEqual(
            {item["paper_id"] for item in ledger["claims"][0]["chunk_snippets"]}, {"1", "2"}
        )

    async def test_retrieval_failure_is_not_insufficient(self):
        await self.seed()
        retrieval = AsyncMock()
        retrieval.search.side_effect = ConnectionError("offline")
        update = await RetrieveEvidenceNode(artifact_store=self.store, content_retrieval=retrieval)(
            self.state
        )
        self.assertEqual(update["error_code"], "RETRIEVAL_FAILED")
        self.assertTrue(update["retryable"])

    async def test_supplemental_retrieval_excludes_seen_and_receives_new_queries(self):
        await self.seed()
        await self.retrieve()
        self.state["revision_items"] = [revision()]
        retrieval = AsyncMock()
        retrieval.search.return_value = [document("2", "new-chunk", "A counterexample.")]
        update = await RetrieveEvidenceNode(artifact_store=self.store, content_retrieval=retrieval)(
            self.state
        )
        inputs = retrieval.search.call_args_list
        self.assertTrue(inputs[0].kwargs["exclude_chunks"])
        self.assertIn("conditional outcomes", [call.kwargs["query"] for call in inputs])
        ledger = await self.store.read_json_uri(update["evidence_ledger_artifact_ref"])
        self.assertEqual(len(ledger["claims"][0]["chunk_snippets"]), 3)

    async def test_opposing_or_insufficient_evidence_routes_to_revision(self):
        for status in ("mixed", "contradicted", "insufficient"):
            await self.seed()
            await self.retrieve()
            self.state.pop("claim_verification_artifact_ref", None)
            self.state.pop("revision_before", None)
            self.state["reflection_round"] = 0
            update = await self.verify(verdict(status))
            self.assertEqual(update["stage"], "retrieving_evidence")
            self.assertEqual(update["reflection_round"], 1)

    async def test_unlocatable_quote_rejected(self):
        await self.seed()
        await self.retrieve()
        result = verdict()
        result["evidence"][0]["quote"] = "Invented improvement"
        update = await self.verify(result)
        self.assertEqual(update["error_code"], "VERIFICATION_FAILED")

    async def test_old_verdict_cannot_support_changed_claim(self):
        await self.seed()
        await self.retrieve()
        await self.verify()
        result = (await self.store.read_json_uri(self.state["claim_verification_artifact_ref"]))[
            "claims"
        ][0]
        with self.assertRaises(ValueError):
            verified_pack(dict(CLAIM, text="A different result is universally better."), result)

    async def test_claim_revision_preserves_identity_and_receives_requirements(self):
        await self.seed()
        await self.retrieve()
        revised = dict(CANDIDATE, text="Outcomes differ under the reported study conditions.")
        model = Model(revised)
        self.state["revision_items"] = [revision("claim")]
        update = await GenerateClaimsNode(
            artifact_store=self.store, model=Model(), revision_model=model
        )(self.state)
        payload = await self.store.read_json_uri(update["claims_artifact_ref"])
        self.assertEqual(payload["claims"][0]["claim_id"], CLAIM["claim_id"])
        self.assertEqual(payload["claims"][0]["question_id"], CLAIM["question_id"])
        self.assertNotEqual(payload["claims"][0]["claim_hash"], CLAIM["claim_hash"])
        self.assertEqual(model.inputs[0]["revisions"], self.state["revision_items"])

    async def test_final_package_requires_two_supporting_papers(self):
        await self.seed()
        await self.retrieve()
        result = verdict()
        result["evidence"][1]["relation"] = "opposes"
        update = await self.verify(result)
        self.assertEqual(update["error_code"], "VERIFICATION_FAILED")

    async def test_large_evidence_is_inspected_in_batches(self):
        await self.seed()
        await self.retrieve(
            [document(text="The result is conditional." + "x" * 30000), document("2", "chunk2")]
        )
        selection = Model(*[dict(evidence=[], limitations="Unrelated window") for _ in range(10)])
        node = VerifyClaimsNode(
            artifact_store=self.store,
            model=Model(verdict("insufficient")),
            selection_model=selection,
        )
        await node(self.state)
        self.assertGreater(len(selection.inputs), 1)
        self.assertTrue(
            all(len(str(item["original_evidence"])) < 24000 for item in selection.inputs)
        )

    async def test_revision_budget_and_no_progress(self):
        await self.seed()
        first = await schedule_revision(
            self.store, self.state, [revision()], claim_ids=["claim_001"], section_ids=[]
        )
        second = await schedule_revision(
            self.store,
            {**self.state, **first},
            [revision()],
            claim_ids=["claim_001"],
            section_ids=[],
        )
        self.assertEqual(second["error_code"], "REVIEW_NO_PROGRESS")
        last = await schedule_revision(
            self.store,
            {**self.state, "reflection_round": 2},
            [revision()],
            claim_ids=["claim_001"],
            section_ids=[],
        )
        self.assertEqual(last["error_code"], "REVIEW_QUALITY_FAILED")

    async def test_writing_and_reflection_use_original_evidence(self):
        await self.seed()
        await self.retrieve()
        self.assertEqual((await self.verify())["stage"], "generating_framework")
        outline = dict(
            title=FOCUS["title"],
            scope=FOCUS["scope"],
            sections=[
                dict(
                    section_id="results",
                    title="Results",
                    description="Conditional outcomes",
                    claim_ids=["claim_001"],
                )
            ],
        )
        self.state.update(
            await GenerateFrameworkNode(artifact_store=self.store, model=Model(outline))(self.state)
        )
        check = (await self.store.read_json_uri(self.state["claim_verification_artifact_ref"]))[
            "claims"
        ][0]
        model = Model(
            dict(
                arguments=[
                    dict(
                        text="Results depend on conditions [[REF_1]] [[REF_2]].",
                        claim_ids=["claim_001"],
                        evidence_ids=[item["evidence_id"] for item in check["evidence"]],
                    )
                ],
                summary="Conditional outcomes",
            )
        )
        update = await RenderSectionsNode(artifact_store=self.store, model=model)(self.state)
        self.assertEqual(update["stage"], "assembling_review", update)
        self.assertEqual(model.inputs[0]["output_language"], "en")
        self.assertEqual(len(model.inputs[0]["claims_with_evidence"][0]["evidence"]), 2)
        self.state.update(update)
        self.state.update(
            await AssembleReviewNode(
                artifact_store=self.store,
                model=Model(
                    dict(abstract="Conditional outcomes.", conclusion="Conditions matter.")
                ),
            )(self.state)
        )
        reflection = Model(
            *[dict(satisfied=True, summary="Supported", issues=[], revisions=[]) for _ in range(2)]
        )
        update = await ReflectReviewNode(artifact_store=self.store, model=reflection)(self.state)
        self.assertEqual(update["stage"], "finalizing_handoff", update)
        self.assertIn("original_evidence", reflection.inputs[0])
        self.assertIn("abstract", reflection.inputs[-1])
        self.state["revision_items"] = [revision("section", "results")]
        rewrite = (
            Model(model.responses[0])
            if model.responses
            else Model(
                dict(
                    arguments=[
                        dict(
                            text="Conditional results [[REF_1]] [[REF_2]].",
                            claim_ids=["claim_001"],
                            evidence_ids=[item["evidence_id"] for item in check["evidence"]],
                        )
                    ],
                    summary="Revised",
                )
            )
        )
        await RenderSectionsNode(artifact_store=self.store, model=rewrite)(self.state)
        self.assertIsNotNone(rewrite.inputs[0]["old_draft"])
        self.assertTrue(rewrite.inputs[0]["revisions"])
        draft = await self.store.read_json_uri(self.state["review_draft_artifact_ref"])
        draft["body_markdown"] += "\nAn unsupported universal improvement."
        draft["conclusion"] = "This proves universal superiority."
        self.state["review_draft_artifact_ref"] = await save(self.store, self.state, "draft", draft)
        reject = Model(
            dict(satisfied=True, summary="Section supported"),
            dict(
                satisfied=False,
                summary="Expanded conclusions",
                issues=[
                    dict(
                        severity="major", category="unsupported", description="New universal claim"
                    )
                ],
                revisions=[revision("section", "results"), revision("synthesis", "review")],
            ),
        )
        result = await ReflectReviewNode(artifact_store=self.store, model=reject)(self.state)
        self.assertEqual(result["stage"], "rendering_sections")
        self.assertEqual(result["reflection_round"], 1)
        self.assertIn("unsupported universal improvement", str(reject.inputs[-1]["body_sections"]))
        self.assertEqual(reject.inputs[-1]["conclusion"], "This proves universal superiority.")

    async def test_real_claim_revision_is_retrieved_and_reverified(self):
        await self.seed()
        await self.retrieve()
        first = verdict("mixed")
        first["revisions"] = [revision("claim")]
        self.assertEqual((await self.verify(first))["stage"], "generating_claims")
        revised = dict(CANDIDATE, text="Outcomes differ only under the reported conditions.")
        self.state.update(
            await GenerateClaimsNode(
                artifact_store=self.store, model=Model(), revision_model=Model(revised)
            )(self.state)
        )
        await self.retrieve()
        result = await self.verify()
        self.assertEqual(result["stage"], "generating_framework", result)
        self.assertEqual(self.state["reflection_round"], 1)

    async def test_initial_claims_receive_full_profiles_and_program_ids(self):
        await self.seed()
        self.state.pop("claims_artifact_ref")
        model = Model(dict(claims=[CANDIDATE]))
        update = await GenerateClaimsNode(
            artifact_store=self.store, model=model, revision_model=Model()
        )(self.state)
        payload = await self.store.read_json_uri(update["claims_artifact_ref"])
        self.assertEqual(payload["claims"][0]["claim_id"], "claim_001")
        self.assertEqual(payload["claims"][0]["question_id"], "rq_001")
        self.assertNotIn("section_id", payload["claims"][0])
        self.assertEqual(model.inputs[0]["studies"][0]["main_discussion"], PROFILE.main_discussion)

    async def test_failure_exposes_evidence_and_draft(self):
        await self.seed()
        await self.retrieve()
        update = await finalize_task_review_node(
            {
                **self.state,
                "stage": "failed",
                "error_code": "REVIEW_QUALITY_FAILED",
                "active_tool_call": {"id": "call", "name": "task_review"},
            }
        )
        result = update["last_action_result"]
        self.assertIn(self.state["evidence_ledger_artifact_ref"], result["artifact_refs"])

    async def test_workflow_routes_new_stages_and_one_revision(self):
        order = []
        counts = {}
        transitions = dict(
            initialize="loading_corpus",
            load_task_corpus="extracting_studies",
            extract_studies="resolving_review_focus",
            resolve_review_focus="generating_claims",
            generate_claims="retrieving_evidence",
            retrieve_evidence="verifying_claims",
            verify_claims="generating_framework",
            generate_framework="rendering_sections",
            render_sections="assembling_review",
            assemble_review="reflecting_review",
            reflect_review="finalizing_handoff",
            finalizing_handoff="persisting_review",
            persist_review="completed",
            finalize_result="completed",
        )

        def node(name):
            async def run(state):
                order.append(name)
                counts[name] = counts.get(name, 0) + 1
                stage = transitions[name]
                if name == "verify_claims" and counts[name] == 1:
                    stage = "generating_claims"
                return {"stage": stage}

            return run

        graph = build_task_review_workflow(
            node_overrides={name: node(name) for name in transitions}
        )
        result = await graph.ainvoke({})
        self.assertEqual(result["stage"], "completed")
        self.assertEqual(counts["verify_claims"], 2)
        self.assertEqual(counts["extract_studies"], 1)
        self.assertLess(order.index("verify_claims"), order.index("generate_framework"))


if __name__ == "__main__":
    unittest.main()
