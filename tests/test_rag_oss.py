import unittest
from uuid import uuid4
from unittest.mock import AsyncMock, patch

from langchain_core.documents import Document
from pydantic import ValidationError

from app.config import settings
from app.infrastructure.grpc.rag_grpc_client import RagGrpcClient
from app.infrastructure.grpc.task_service_grpc_client import TaskState
from app.infrastructure.mineru.mineru_client import MinerUClient
from app.llm.graph.workflows.task_indexing.nodes.run_or_wait import (
    RunOrWaitTaskRagNode,
)
from app.llm.subagents.task_indexing import TaskIndexingDelegation
from app.protos.rag.v1 import rag_pb2
from app.rag.data_preparation.data_preparation import DataPreparationModule
from app.rag.file_parser.file_parser import FileParser
from app.rag.processing.paper_rag_processor import (
    PaperRagInput,
    PaperRagProcessor,
    PaperRagResult,
)
from app.rag.processing.task_rag_batch_runner import TaskRagBatchRunner
from app.schemas.file_parser import (
    BatchDocumentParseResult,
    FileParseRequest,
    FileType,
    ParsedFileDocument,
)


SHA256 = "a" * 64
SOURCE_URL = "https://example.test/paper.pdf"
OSS_URL = "https://bucket.example.test/paper.pdf?signature=secret"


class _Parser:
    def __init__(self) -> None:
        self.requests = []

    async def parse_files_result(self, requests):
        self.requests = requests
        return BatchDocumentParseResult(
            total=len(requests),
            documents=[
                ParsedFileDocument(
                    document=Document(
                        page_content="Parsed markdown.",
                        metadata={"file_id": request.file_id},
                    )
                )
                for request in requests
            ],
            errors={},
        )


class _Index:
    async def build_index(self, documents) -> None:
        return None


class _OssStore:
    def __init__(self, *, pdf_url: str | None, fails: bool = False) -> None:
        self.pdf_url = pdf_url
        self.fails = fails
        self.requested_sha256 = []
        self.markdown_calls = []

    async def get_pdf_temporary_url(self, *, pdf_sha256: str) -> str | None:
        self.requested_sha256.append(pdf_sha256)
        if self.fails:
            raise RuntimeError("OSS unavailable")
        return self.pdf_url

    async def upload_markdown(self, *, markdown: str, pdf_sha256: str) -> None:
        self.markdown_calls.append((markdown, pdf_sha256))


def _processor(parser: _Parser, oss_store: _OssStore) -> PaperRagProcessor:
    return PaperRagProcessor(
        file_parser=parser,
        data_preparation=DataPreparationModule(),
        paper_index=_Index(),
        paper_content_index=_Index(),
        oss_store=oss_store,
    )


class PaperRagOssTests(unittest.IsolatedAsyncioTestCase):
    async def test_prefers_oss_url_and_saves_markdown(self) -> None:
        parser = _Parser()
        oss_store = _OssStore(pdf_url=OSS_URL)

        results = await _processor(parser, oss_store).process_many(
            [
                PaperRagInput(
                    paper_id=1,
                    title="Paper",
                    abstract="Abstract",
                    pdf_url=SOURCE_URL,
                    oss_name=SHA256,
                )
            ]
        )

        self.assertEqual(str(parser.requests[0].uri), OSS_URL)
        self.assertEqual(parser.requests[0].metadata["source"], SOURCE_URL)
        self.assertEqual(oss_store.requested_sha256, [SHA256])
        self.assertEqual(oss_store.markdown_calls, [("Parsed markdown.", SHA256)])
        self.assertEqual(results[0].status, "ready")

    async def test_falls_back_to_source_url_when_oss_is_unavailable(self) -> None:
        parser = _Parser()
        oss_store = _OssStore(pdf_url=None, fails=True)

        await _processor(parser, oss_store).process_many(
            [
                PaperRagInput(
                    paper_id=1,
                    title="Paper",
                    abstract="Abstract",
                    pdf_url=SOURCE_URL,
                    oss_name=SHA256,
                )
            ]
        )

        self.assertEqual(str(parser.requests[0].uri), SOURCE_URL)
        self.assertEqual(oss_store.markdown_calls, [("Parsed markdown.", SHA256)])


class _RagStub:
    async def ClaimTaskRagPapers(self, request, *, timeout):
        return rag_pb2.ClaimTaskRagPapersResponse(
            success=True,
            task_id=request.task_id,
            batch_id=request.batch_id,
            papers=[
                rag_pb2.PaperRagInput(
                    paper_id=1,
                    title="Paper",
                    url=SOURCE_URL,
                    paper_abstract="Abstract",
                    oss_name=SHA256,
                )
            ],
        )


class _RagClient(RagGrpcClient):
    async def _get_stub(self):
        return _RagStub()


class RagGrpcClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_claim_preserves_oss_name(self) -> None:
        result = await _RagClient().claim_task_rag_papers(
            task_id=1,
            batch_id=str(uuid4()),
            limit=1,
            lease_seconds=60,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["papers"][0]["oss_name"], SHA256)


class _MinerU:
    async def download_result_zip(self, url: str) -> bytes:
        return b"zip"

    def extract_result_files(self, content: bytes) -> dict:
        return {}

    def extract_markdown(self, result_files: dict) -> str:
        return "Parsed markdown."

    def extract_content_list(self, result_files: dict):
        return None


class FileParserSourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_safe_source_metadata_instead_of_presigned_url(self) -> None:
        result = await FileParser(mineru_client=_MinerU())._build_document(
            request=FileParseRequest(
                file_id="paper-1",
                uri=OSS_URL,
                file_name="paper-1.pdf",
                file_type=FileType.PDF,
                metadata={"source": SOURCE_URL},
            ),
            result={"full_zip_url": "https://example.test/result.zip"},
            model_version="vlm",
        )

        self.assertEqual(result.document.metadata["source"], SOURCE_URL)


class MinerUBatchLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_logs_submission_and_batch_result_progress_without_urls(self) -> None:
        client = MinerUClient(token="test-token")
        client._request_json = AsyncMock(
            side_effect=[
                {"data": {"batch_id": "mineru-batch-1"}},
                {
                    "data": {
                        "extract_result": [
                            {
                                "data_id": "paper-7",
                                "state": "running",
                                "extract_progress": {
                                    "extracted_pages": 2,
                                    "total_pages": 8,
                                },
                            }
                        ]
                    }
                },
                {
                    "data": {
                        "extract_result": [
                            {
                                "data_id": "paper-7",
                                "state": "done",
                                "full_zip_url": (
                                    "https://bucket.example.test/result.zip?signature=secret"
                                ),
                            }
                        ]
                    }
                },
            ]
        )

        try:
            with (
                self.assertLogs(
                    "app.infrastructure.mineru.mineru_client",
                    level="INFO",
                ) as logs,
                patch(
                    "app.infrastructure.mineru.mineru_client.asyncio.sleep",
                    new=AsyncMock(),
                ),
            ):
                batch_id = await client.submit_batch_urls(
                    files=[
                        {
                            "url": (
                                "https://bucket.example.test/paper.pdf?signature=secret"
                            ),
                            "data_id": "paper-7",
                        }
                    ],
                )
                results = await client.wait_batch(batch_id)
        finally:
            await client.close()

        output = "\n".join(logs.output)
        self.assertEqual(batch_id, "mineru-batch-1")
        self.assertEqual(results[0]["state"], "done")
        self.assertIn("MinerU batch submitted", output)
        self.assertIn("MinerU batch result update", output)
        self.assertIn("MinerU batch completed", output)
        self.assertIn("paper-7", output)
        self.assertNotIn("signature=secret", output)


class TaskIndexingTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_defaults_leave_post_processing_time_after_mineru(self) -> None:
        request = TaskIndexingDelegation(task_id=1)
        client = MinerUClient(token="test-token")
        try:
            self.assertEqual(
                request.timeout_seconds,
                settings.TASK_RAG_WORKFLOW_TIMEOUT_SECONDS,
            )
            self.assertEqual(
                client.parse_timeout,
                settings.MINERU_PARSE_TIMEOUT_SECONDS,
            )
        finally:
            await client.close()

    async def test_rejects_timeout_without_post_processing_budget(self) -> None:
        minimum = (
            settings.MINERU_PARSE_TIMEOUT_SECONDS
            + settings.TASK_RAG_POST_PROCESSING_GRACE_SECONDS
        )
        with self.assertRaises(ValidationError):
            TaskIndexingDelegation(task_id=1, timeout_seconds=minimum - 1)

    async def test_timeout_reconciles_remote_terminal_status(self) -> None:
        cases = (
            ("RAG_FAILED", False, "failed", "RAG_INDEX_FAILED"),
            ("RAG_COMPLETED", True, "verifying", None),
            ("RAG_RUNNING", False, "timed_out", "RAG_INDEX_TIMEOUT"),
        )
        for task_status, complete, expected_stage, expected_error in cases:
            with self.subTest(task_status=task_status):
                node = RunOrWaitTaskRagNode(runner=_TimeoutRunner())
                node._wait_for_result_or_publication_failure = AsyncMock(
                    side_effect=TimeoutError,
                )
                status = _rag_status(
                    task_status=task_status,
                    is_rag_complete=complete,
                )
                with (
                    patch(
                        "app.llm.graph.workflows.task_indexing.nodes.run_or_wait"
                        ".current_event_context",
                        return_value=_EventContext(),
                    ),
                    patch(
                        "app.llm.graph.workflows.task_indexing.nodes.run_or_wait"
                        ".get_task_rag_status",
                        new=AsyncMock(return_value=status),
                    ),
                ):
                    update = await node(
                        {
                            "task_id": 1,
                            "remote_task_state": TaskState.SEARCH_COMPLETED.name,
                            "overall_summary": {},
                            "timeout_seconds": 1,
                        }
                    )

                self.assertEqual(update["stage"], expected_stage)
                self.assertEqual(update.get("error_code"), expected_error)


class TaskRagLoggingTests(unittest.TestCase):
    def test_failed_paper_log_redacts_urls(self) -> None:
        with self.assertLogs(
            "app.rag.processing.task_rag_batch_runner",
            level="WARNING",
        ) as logs:
            TaskRagBatchRunner._log_paper_results(
                task_id=23,
                batch_id="batch-1",
                results=[
                    PaperRagResult(
                        paper_id=7,
                        status="failed",
                        error=(
                            "MinerUError: download failed at "
                            "https://bucket.example.test/file.pdf?signature=secret"
                        ),
                    )
                ],
            )

        output = "\n".join(logs.output)
        self.assertIn("task_id=23", output)
        self.assertIn("paper_id=7", output)
        self.assertIn("<url>", output)
        self.assertNotIn("signature=secret", output)


class _TimeoutHandle:
    async def unsubscribe(self) -> None:
        return None


class _TimeoutRunner:
    async def start_or_join(self, *args, **kwargs):
        return _TimeoutHandle()


class _EventContext:
    def scoped(self, **kwargs):
        return self


def _rag_status(*, task_status: str, is_rag_complete: bool) -> dict:
    return {
        "ok": True,
        "result": {
            "task_id": 1,
            "task_status": task_status,
            "summary": {
                "total": 1,
                "pending": 0,
                "indexing": 0,
                "ready": 0,
                "skipped": 0,
                "failed": 1 if task_status == "RAG_FAILED" else 0,
                "ready_chunks": 0,
            },
            "is_rag_complete": is_rag_complete,
            "can_generate_review": is_rag_complete,
        },
        "error": None,
    }


if __name__ == "__main__":
    unittest.main()
