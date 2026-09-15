import json
import tempfile
import unittest

from langchain_core.documents import Document

from app.infrastructure.oss import MarkdownObject
from app.infrastructure.grpc.paper_service_grpc_client import (
    PaperServiceGrpcClient,
)
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.nodes.extract_studies import (
    EVIDENCE_MAP_MAX_CHARS,
    ExtractStudiesNode,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import (
    finalize_task_review_node,
)
from app.llm.graph.workflows.review_generate.nodes.load import (
    LoadTaskCorpusNode,
)
from app.protos.paper.v1 import paper_pb2


SHA256 = "a" * 64
FULLTEXT = """# Methods
This randomized trial enrolled 40 adults.

# Results
The intervention reduced symptoms.

# Limitations
The sample was small.
"""


class _OssStore:
    def __init__(self, markdown: str | None) -> None:
        self.markdown = markdown

    async def get_markdown(self, *, pdf_sha256: str):
        if self.markdown is None:
            return None
        return MarkdownObject(
            content=self.markdown,
            object_name=f"papers/md/{pdf_sha256}.md",
            content_sha256="b" * 64,
        )


class _CorpusReader:
    async def read(self, paper_ids):
        return [
            Document(
                page_content=FULLTEXT,
                metadata={"chunk_id": "chunk-1"},
            )
        ]


class _Model:
    async def ainvoke(self, _messages):
        return {
            "design": {
                "summary": "randomized trial",
                "evidence_quotes": ["This randomized trial enrolled 40 adults."],
            },
            "sample": {
                "summary": "40 adults",
                "evidence_quotes": ["This randomized trial enrolled 40 adults."],
            },
            "methods": {
                "summary": "not_reported",
                "evidence_quotes": [],
            },
            "results": {
                "summary": "reduced symptoms",
                "evidence_quotes": ["The intervention reduced symptoms."],
            },
            "limitations": {
                "summary": "small sample",
                "evidence_quotes": ["The sample was small."],
            },
        }


class _FailingModel:
    async def ainvoke(self, _messages):
        raise RuntimeError("model unavailable")


class _PaperClient:
    async def get_task_review_papers(self, task_id: int) -> dict:
        return {
            "ok": True,
            "result": {
                "task_id": task_id,
                "task_status": "rag_completed",
                "papers": [
                    {
                        "paper_id": 1,
                        "title": "Paper",
                        "authors": [],
                        "paper_abstract": "Abstract",
                        "oss_name": SHA256,
                        "pdf_url": "https://example.test/paper.pdf",
                        "rag_status": "ready",
                        "chunk_count": 1,
                    }
                ],
            },
            "error": None,
        }


class _ReviewPapersStub:
    async def GetTaskReviewPapers(self, request, *, timeout):
        self.task_id = request.task_id
        self.timeout = timeout
        return paper_pb2.GetTaskReviewPapersResponse(
            success=True,
            task_id=request.task_id,
            task_status="RAG_COMPLETED",
            papers=[
                paper_pb2.TaskReviewPaperItem(
                    paper_id=1,
                    title="Paper",
                    paper_abstract=" Abstract ",
                    oss_name=SHA256,
                    pdf_url="https://example.test/paper.pdf",
                    rag_status="READY",
                    chunk_count=1,
                )
            ],
        )


class _PaperServiceClient(PaperServiceGrpcClient):
    def __init__(self, stub: _ReviewPapersStub) -> None:
        self.stub = stub

    async def _get_stub(self) -> _ReviewPapersStub:
        return self.stub


class ExtractStudiesNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_review_papers_rpc_maps_fulltext_sources(self) -> None:
        response = await _PaperServiceClient(_ReviewPapersStub()).get_task_review_papers(1)

        self.assertTrue(response["ok"])
        paper = response["result"]["papers"][0]
        self.assertEqual(paper["paper_abstract"], "Abstract")
        self.assertEqual(paper["oss_name"], SHA256)
        self.assertEqual(paper["pdf_url"], "https://example.test/paper.pdf")

    async def test_load_corpus_preserves_fulltext_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            update = await LoadTaskCorpusNode(
                client=_PaperClient(),
                artifact_store=store,
            )(
                {
                    "stage": "loading_corpus",
                    "run_id": "review-test",
                    "task_id": 1,
                    "warnings": [],
                }
            )

            payload = await store.read_json_uri(update["corpus_artifact_ref"])
            paper = payload["papers"][0]
            self.assertEqual(update["stage"], "extracting_studies")
            self.assertEqual(paper["abstract"], "Abstract")
            self.assertEqual(paper["oss_name"], SHA256)
            self.assertEqual(paper["pdf_url"], "https://example.test/paper.pdf")

    async def _run(self, markdown: str | None) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            corpus = await store.write_json(
                run_id="review-test",
                step_key="load_task_corpus",
                source="test",
                kind="corpus",
                payload={
                    "papers": [
                        {
                            "paper_id": 1,
                            "title": "Paper",
                            "oss_name": SHA256,
                            "rag_status": "ready",
                            "chunk_count": 1,
                        }
                    ]
                },
            )
            node = ExtractStudiesNode(
                artifact_store=store,
                model=_Model(),
                oss_store=_OssStore(markdown),
                corpus_reader=_CorpusReader(),
            )

            update = await node(
                {
                    "stage": "extracting_studies",
                    "run_id": "review-test",
                    "task_id": 1,
                    "topic": "Topic",
                    "language": "English",
                    "paper_ids_snapshot": ["1"],
                    "corpus_artifact_ref": corpus.artifact_uri,
                    "warnings": [],
                }
            )
            self.assertEqual(update["stage"], "generating_framework")
            return await store.read_json_uri(update["study_records_artifact_ref"])

    async def test_prefers_oss_markdown(self) -> None:
        payload = await self._run(FULLTEXT)

        study = payload["studies"][0]
        self.assertEqual(study["source"]["kind"], "oss_markdown")
        self.assertEqual(study["results"]["summary"], "reduced symptoms")

    async def test_falls_back_to_qdrant_chunks(self) -> None:
        payload = await self._run(None)

        study = payload["studies"][0]
        self.assertEqual(study["source"]["kind"], "qdrant_chunk_fallback")
        self.assertEqual(study["source"]["chunk_ids"], ["chunk-1"])

    def test_evidence_map_is_bounded_and_keeps_paper_ids(self) -> None:
        records = [
            {
                "paper_id": str(paper_id),
                "title": "Paper",
                **{
                    field_name: {"summary": "x" * 1200}
                    for field_name in (
                        "design",
                        "sample",
                        "methods",
                        "results",
                        "limitations",
                    )
                },
            }
            for paper_id in range(1, 51)
        ]

        evidence_map = ExtractStudiesNode._evidence_map(records)

        self.assertLessEqual(
            len(json.dumps(evidence_map, ensure_ascii=False, separators=(",", ":"))),
            EVIDENCE_MAP_MAX_CHARS,
        )
        self.assertEqual(
            [item["paper_id"] for item in evidence_map],
            [str(paper_id) for paper_id in range(1, 51)],
        )

    async def test_failure_blocks_review_and_returns_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            corpus = await store.write_json(
                run_id="review-test",
                step_key="load_task_corpus",
                source="test",
                kind="corpus",
                payload={
                    "papers": [
                        {
                            "paper_id": 1,
                            "title": "Paper",
                            "oss_name": SHA256,
                        }
                    ]
                },
            )
            update = await ExtractStudiesNode(
                artifact_store=store,
                model=_FailingModel(),
                oss_store=_OssStore(FULLTEXT),
                corpus_reader=_CorpusReader(),
            )(
                {
                    "stage": "extracting_studies",
                    "run_id": "review-test",
                    "task_id": 1,
                    "topic": "Topic",
                    "language": "English",
                    "paper_ids_snapshot": ["1"],
                    "corpus_artifact_ref": corpus.artifact_uri,
                    "warnings": [],
                }
            )

            self.assertEqual(update["stage"], "failed")
            report_ref = update["study_extraction_report_artifact_ref"]
            report = await store.read_json_uri(report_ref)
            self.assertEqual(report["extracted_paper_ids"], [])
            self.assertEqual(report["failures"][0]["paper_id"], "1")

            result = await finalize_task_review_node(
                {
                    **update,
                    "active_tool_call": {
                        "id": "call-1",
                        "name": "task_review",
                    },
                    "task_id": 1,
                }
            )
            action = result["last_action_result"]
            self.assertIn(report_ref, action["artifact_refs"])
            self.assertEqual(
                action["data"]["study_extraction_report_artifact_ref"],
                report_ref,
            )


if __name__ == "__main__":
    unittest.main()
