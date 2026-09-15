import unittest
from uuid import uuid4

from langchain_core.documents import Document

from app.infrastructure.grpc.rag_grpc_client import RagGrpcClient
from app.protos.rag.v1 import rag_pb2
from app.rag.data_preparation.data_preparation import DataPreparationModule
from app.rag.file_parser.file_parser import FileParser
from app.rag.processing.paper_rag_processor import (
    PaperRagInput,
    PaperRagProcessor,
)
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


if __name__ == "__main__":
    unittest.main()
