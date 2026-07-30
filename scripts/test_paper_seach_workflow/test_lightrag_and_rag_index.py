import tempfile
import unittest
from pathlib import Path

import httpx

from app.infrastructure.lightrag_client import LightRAGClient
from app.llm.artifacts.store import LocalArtifactStore
from app.services.rag_index_runner import RagIndexRunner


class LightRAGClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_upload_returns_track_id_and_uses_api_key(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["X-API-Key"], "test-key")
            self.assertEqual(request.url.path, "/documents/upload")
            return httpx.Response(200, json={"data": {"track_id": "track-1"}})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "paper.pdf"
                path.write_bytes(b"%PDF-test")
                client = LightRAGClient(
                    base_url="http://lightrag.test",
                    api_key="test-key",
                    client=http_client,
                )
                track_id = await client.upload_file(path)

        self.assertEqual(track_id, "track-1")


class RagIndexRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_uploads_pdf_polls_until_processed_and_writes_result(self) -> None:
        class FakeLightRAGClient:
            def __init__(self) -> None:
                self.statuses = iter([{"status": "processing"}, {"status": "processed"}])
                self.uploaded_paths: list[str] = []

            async def upload_file(self, path: str) -> str:
                self.uploaded_paths.append(path)
                return "track-101"

            async def get_track_status(self, track_id: str) -> dict:
                if track_id != "track-101":
                    raise AssertionError("unexpected track ID")
                return next(self.statuses)

        class FakePaperRagIndexClient:
            def __init__(self) -> None:
                self.completed: list[dict] = []

            async def ensure(self, items: list[dict]) -> dict:
                return {
                    "ok": True,
                    "result": {
                        "indexes": [{
                            "index": {
                                "paper_id": items[0]["paper_id"],
                                "content_hash": items[0]["content_hash"],
                                "index_status": "pending",
                                "lightrag_document_id": None,
                                "lightrag_track_id": None,
                                "error_message": None,
                            },
                            "should_index": True,
                        }]
                    },
                    "error": None,
                }

            async def claim(self, items: list[dict], **_kwargs) -> dict:
                return {
                    "ok": True,
                    "result": {"claimed": [{"paper_id": items[0]["paper_id"], "lease_token": "lease-101"}]},
                    "error": None,
                }

            async def complete(self, **kwargs) -> dict:
                self.completed.append(kwargs)
                return {"ok": True, "result": {"accepted": True}, "error": None}

            async def skip(self, _paper_ids: list[int], *, reason: str) -> dict:
                return {"ok": True, "result": {"skipped_count": 1}, "error": None}

        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            pdf_path = Path(directory) / "paper.pdf"
            pdf_path.write_bytes(b"%PDF-test")
            request = await store.write_json(
                run_id="run-1",
                step_key="rag_index_request",
                source="lightrag",
                kind="test",
                payload={
                    "run_id": "run-1",
                    "search_task_id": 7,
                    "requests": [{"paper_id": 101, "source_pdf_path": str(pdf_path)}],
                },
            )
            fake_client = FakeLightRAGClient()
            fake_rag_index_client = FakePaperRagIndexClient()
            runner = RagIndexRunner(
                artifact_store=store,
                lightrag_client=fake_client,
                rag_index_client=fake_rag_index_client,
                poll_interval_seconds=0,
                max_polls=2,
            )
            result_uri = await runner.run(request.artifact_uri)
            result_path = store.base_dir / result_uri.removeprefix("artifact://")
            result = __import__("json").loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(fake_client.uploaded_paths, [str(pdf_path)])
        self.assertEqual(result["results"][0]["status"], "ready")
        self.assertEqual(result["results"][0]["track_id"], "track-101")
        self.assertEqual(fake_rag_index_client.completed[0]["index_status"], "ready")


if __name__ == "__main__":
    unittest.main()
