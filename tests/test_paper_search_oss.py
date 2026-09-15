import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import settings
from app.infrastructure.oss import (
    build_markdown_object_name,
    build_pdf_object_name,
)
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.paper_search.nodes.paper_cache import (
    PaperCacheStore,
)
from app.llm.graph.workflows.paper_search.nodes.persist import (
    PersistRecommendedPapersNode,
)
from app.llm.graph.workflows.paper_search.nodes.save_oss import (
    SavePersistedPdfsToOssNode,
)
from app.llm.graph.workflows.paper_search.nodes.venue import VenueCacheStore


RUN_ID = "oss-save-test"
TASK_ID = 7
SHA256 = "a" * 64
PDF_OBJECT_NAME = f"papers/pdf/{SHA256}.pdf"
MARKDOWN_OBJECT_NAME = f"papers/md/{SHA256}.md"


class FakeOssObjectStore:
    def __init__(self, failing_names: set[str] | None = None) -> None:
        self.calls: list[tuple[Path, str, str]] = []
        self.failing_names = failing_names or set()

    async def upload_pdf(
        self,
        *,
        source_path: Path,
        object_name: str,
        sha256: str,
    ) -> None:
        self.calls.append((source_path, object_name, sha256))
        if object_name in self.failing_names:
            raise RuntimeError("simulated OSS failure")


def _paper(pdf_path: Path, sha256: str = SHA256) -> dict:
    return {
        "paper_id": None,
        "paper_info": {
            "title": "A paper about OSS",
            "authors": ["Ada Lovelace"],
            "doi": "10.1000/oss-test",
            "pdf_url": "https://example.test/paper.pdf",
        },
        "pdf_download": {
            "status": "success",
            "local_pdf_path": str(pdf_path),
            "sha256": sha256,
        },
        "task_paper_relation_draft": {
            "search_task_id": TASK_ID,
            "paper_id": None,
            "recommendation_stars": 4,
            "recommendation_reason": "Relevant to the requested topic.",
        },
    }


class PaperSearchOssTests(unittest.IsolatedAsyncioTestCase):
    async def test_persist_fallback_manifest_uploads_pdf_to_oss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            pdf_path = store.base_dir / RUN_ID / "pdfs" / "paper.pdf"
            pdf_path.parent.mkdir(parents=True)
            pdf_path.write_bytes(b"%PDF-1.4")
            recommendation_artifact = await store.write_json(
                run_id=RUN_ID,
                step_key="bind_task",
                source="test",
                kind="recommendation_manifest_json",
                payload={"papers": [_paper(pdf_path)]},
            )

            saved_requests: list[dict] = []

            async def save_papers(requests: list[dict]) -> dict:
                saved_requests.extend(requests)
                return {
                    "ok": True,
                    "result": {
                        "papers": [
                            {
                                "client_key": request["client_key"],
                                "paper_id": 42,
                            }
                            for request in requests
                        ]
                    },
                }

            async def save_relations(requests: list[dict]) -> dict:
                return {"ok": True, "result": {"saved_count": len(requests)}}

            persist = PersistRecommendedPapersNode(
                artifact_store=store,
                save_papers=save_papers,
                save_task_papers=save_relations,
                venue_cache_store=VenueCacheStore(store.base_dir / "venues.json"),
                paper_cache_store=PaperCacheStore(store.base_dir / "papers.json"),
            )
            object_store = FakeOssObjectStore()

            with patch.object(settings, "OSS_ENABLED", True), patch.object(
                settings, "OSS_PREFIX", "papers"
            ):
                update = await persist(
                    {
                        "run_id": RUN_ID,
                        "paper_service_task_id": TASK_ID,
                        "recommendation_manifest_artifact_ref": (
                            recommendation_artifact.artifact_uri
                        ),
                    }
                )
                await SavePersistedPdfsToOssNode(
                    artifact_store=store,
                    object_store=object_store,
                )(
                    {
                        "run_id": RUN_ID,
                        "persisted_papers_manifest_artifact_ref": update[
                            "persisted_papers_manifest_artifact_ref"
                        ],
                        "recommendation_manifest_artifact_ref": (
                            recommendation_artifact.artifact_uri
                        ),
                    }
                )

            self.assertEqual(
                saved_requests[0]["paper_info"]["oss_name"],
                SHA256,
            )
            persisted = await store.read_json_uri(
                update["persisted_papers_manifest_artifact_ref"]
            )
            self.assertEqual(persisted["results"][0]["oss_name"], SHA256)
            self.assertEqual(
                object_store.calls,
                [(pdf_path.resolve(), PDF_OBJECT_NAME, SHA256)],
            )

    async def test_one_upload_failure_does_not_stop_other_papers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            second_sha256 = "b" * 64
            first_path = store.base_dir / RUN_ID / "pdfs" / "first.pdf"
            second_path = store.base_dir / RUN_ID / "pdfs" / "second.pdf"
            first_path.parent.mkdir(parents=True)
            first_path.write_bytes(b"first")
            second_path.write_bytes(b"second")
            second_pdf_object_name = build_pdf_object_name(
                prefix="papers",
                pdf_sha256=second_sha256,
            )
            self.assertIsNotNone(second_pdf_object_name)

            bound_artifact = await store.write_json(
                run_id=RUN_ID,
                step_key="bind_task",
                source="test",
                kind="recommendation_manifest_json",
                payload={
                    "papers": [
                        _paper(first_path),
                        _paper(second_path, second_sha256),
                    ]
                },
            )
            persisted_artifact = await store.write_json(
                run_id=RUN_ID,
                step_key="persist",
                source="test",
                kind="persisted_papers_manifest_json",
                payload={
                    "results": [
                        {
                            "client_key": f"{RUN_ID}:0",
                            "paper_id": 1,
                            "paper_save_status": "saved",
                            "oss_name": SHA256,
                        },
                        {
                            "client_key": f"{RUN_ID}:1",
                            "paper_id": 2,
                            "paper_save_status": "saved",
                            "oss_name": second_sha256,
                        },
                    ]
                },
            )
            object_store = FakeOssObjectStore({PDF_OBJECT_NAME})

            with patch.object(settings, "OSS_ENABLED", True), patch.object(
                settings, "OSS_PREFIX", "papers"
            ):
                update = await SavePersistedPdfsToOssNode(
                    artifact_store=store,
                    object_store=object_store,
                )(
                    {
                        "run_id": RUN_ID,
                        "persisted_papers_manifest_artifact_ref": (
                            persisted_artifact.artifact_uri
                        ),
                        "task_bound_recommendation_manifest_artifact_ref": (
                            bound_artifact.artifact_uri
                        ),
                    }
                )

            self.assertEqual(update, {})
            self.assertEqual(
                [(call[1], call[2]) for call in object_store.calls],
                [
                    (PDF_OBJECT_NAME, SHA256),
                    (second_pdf_object_name, second_sha256),
                ],
            )

    def test_pdf_and_markdown_keys_share_the_pdf_sha256(self) -> None:
        self.assertEqual(
            build_pdf_object_name(
                prefix="papers/",
                pdf_sha256=SHA256.upper(),
            ),
            PDF_OBJECT_NAME,
        )
        self.assertEqual(
            build_markdown_object_name(
                prefix="papers/",
                pdf_sha256=SHA256.upper(),
            ),
            MARKDOWN_OBJECT_NAME,
        )

    async def test_legacy_oss_object_name_is_not_uploaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            pdf_path = store.base_dir / RUN_ID / "pdfs" / "paper.pdf"
            pdf_path.parent.mkdir(parents=True)
            pdf_path.write_bytes(b"pdf")
            bound_artifact = await store.write_json(
                run_id=RUN_ID,
                step_key="bind_task",
                source="test",
                kind="recommendation_manifest_json",
                payload={"papers": [_paper(pdf_path)]},
            )
            persisted_artifact = await store.write_json(
                run_id=RUN_ID,
                step_key="persist",
                source="test",
                kind="persisted_papers_manifest_json",
                payload={
                    "results": [
                        {
                            "client_key": f"{RUN_ID}:0",
                            "paper_id": 1,
                            "paper_save_status": "saved",
                            "oss_name": PDF_OBJECT_NAME,
                        }
                    ]
                },
            )
            object_store = FakeOssObjectStore()

            with patch.object(settings, "OSS_ENABLED", True), patch.object(
                settings, "OSS_PREFIX", "papers"
            ):
                await SavePersistedPdfsToOssNode(
                    artifact_store=store,
                    object_store=object_store,
                )(
                    {
                        "run_id": RUN_ID,
                        "persisted_papers_manifest_artifact_ref": (
                            persisted_artifact.artifact_uri
                        ),
                        "task_bound_recommendation_manifest_artifact_ref": (
                            bound_artifact.artifact_uri
                        ),
                    }
                )

            self.assertEqual(object_store.calls, [])


if __name__ == "__main__":
    unittest.main()
